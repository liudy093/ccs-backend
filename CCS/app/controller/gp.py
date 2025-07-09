import asyncio
import random
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union, cast

import aiodocker
import aiohttp
from aioredis import Redis
from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_500_INTERNAL_SERVER_ERROR,
    WS_1008_POLICY_VIOLATION,
)

from app.configuration import GPU_MECHINE_IPS, MONGODB_HOST, LOGGING_NAME
from app.depends.jwt_auth import JWTUser, get_current_user
from app.depends.redis import get_redis_connection, get_redis_connection_manu
from app.utils.redlock import RedLock
from starlette.websockets import WebSocket
import logging


router = APIRouter()
logger: logging.Logger = logging.getLogger(LOGGING_NAME).getChild("gp")


class Task(BaseModel):
    id: str
    name: str
    state: str  # 此字段由后端程序根据mechine_id来设定，仅仅为了前端使用方便，对后端没有意义（后端程序逻辑不依赖此字段）
    mechine_id: int  # 任务所执行的机器id, ==-1 没有调度，没有执行；>=0 已经调度，开始执行


class Action(BaseModel):
    action: str
    taskid: str


class AddTaskInfo(BaseModel):
    image: str
    name: str


@dataclass
class TaskInternal:
    id: str
    name: str
    image: str
    container_id: str  # 任务所执行的容器id != -1：已经调度，==-1：没有调度
    mechine_id: int  # 任务所执行的机器id, ==-1 没有调度；>=0 已经调度
    btime: float  # 任务进入等待列队时间点
    rtime: float  # 任务开始运行时间点
    state: str  # running：正在运行，stop：停止运行，queue：等待调度


@dataclass
class MechineInfo:
    id: int
    cpu: float  # cpu 利用率 %
    mem: float  # mem 利用率 %
    gpu: float  # gup 利用率 %
    gpu_mem: float  # gpu mem 利用率 %


# 目标服务器需要启用 docker 的tcp模式，"-H tcp://0.0.0.0:2375"
_GPU_MECHINE_IPS = GPU_MECHINE_IPS.split(",")
GPU_MECHINES = ["http://{}:2375".format(ip) for ip in _GPU_MECHINE_IPS]


def schedule_task(
    mechines: List[MechineInfo], tasks: List[TaskInternal]
) -> List[Tuple[int, TaskInternal]]:
    """
    :param mechines: 所有可供调度的机器信息
    :param tasks: 所有等待调度任务的信息

    :return 调度到的机器id与任务对应关系
    """
    # FCFS 先来先服务
    # 如果gpu利用率不超过 50%，则调度

    rst = []
    for mechine in mechines:
        if mechine.gpu <= 50.0:
            rst.append((mechine.id, tasks[0]))

    return rst


async def start_task(mechine_id: int, task_internal_info: TaskInternal) -> str:
    docker = aiodocker.Docker(GPU_MECHINES[mechine_id])
    container = await docker.containers.create_or_replace(
        config={"Image": task_internal_info.image}, name=task_internal_info.id
    )
    await container.start()
    container_id = container._id
    await docker.close()
    return container_id


async def task_scheduler():
    logger.info("启动任务调度")
    while True:
        rds = await get_redis_connection_manu()
        try:
            logger.debug("随机等待，尝试获得调度锁")
            await asyncio.sleep(random.randint(2, 10))  # 随机等待 2~10 秒，等概率避免多机共同竞争锁
            async with RedLock(rds, "gp_scheduler") as locker:
                if not locker:  # 没能获得锁，继续等待
                    logger.debug("本次未能获得调度锁")
                    continue

                logger.debug("获得调度锁，开始任务调度")

                # 首先要检查正在执行的任务情况
                # 遍历每一台gpu服务器，找到已经停止的容器，
                #   并从运行列队移除，并设置任务的属性
                running_tasks = await rds.smembers("running_tasks")
                if len(running_tasks) > 0:
                    logger.debug("发现运行任务({}), 开始检查所有运行任务状态".format(running_tasks))
                    running_container_ids = [
                        await rds.hget(rt, "container_id", encoding="utf-8")
                        for rt in running_tasks
                    ]
                    for idx, url in enumerate(GPU_MECHINES):
                        logger.debug("连接Docker：{}".format(url))
                        docker = aiodocker.Docker(url)
                        try:
                            containers_data = await docker._query_json(
                                "containers/json",
                                method="GET",
                                params={"all": True},
                                timeout=3,  # 局域网内 3s 超时
                            )
                            logger.debug("此Docker返回以下容器信息：{}".format(containers_data))
                            for d in containers_data:
                                if d["State"] == "Exited":
                                    logger.debug(
                                        "发现已经执行结束的任务（State==Exited）：{}".format(d)
                                    )
                                    container_id = d["Id"]
                                    if container_id in running_container_ids:
                                        task_id = running_tasks[
                                            running_container_ids.index(container_id)
                                        ]
                                        await rds.srem("running_tasks", task_id)
                                        await rds.hmset_dict(task_id, {"state": "stop"})
                                        logger.debug(
                                            "任务{}从运行列队中移除，并设置任务状态为stop（state=stop）"
                                        )
                        except:
                            logger.error("连接Docker获取信息的过程中出错")
                        finally:
                            await docker.close()

                # 获取正在排队的任务信息
                queued_tasks = await rds.smembers("queued_tasks")
                if len(queued_tasks) > 0:
                    tasks: List[TaskInternal] = []
                    for qt in queued_tasks:
                        t = await rds.hgetall(qt, encoding="utf-8")
                        tasks.append(TaskInternal(**t))
                    logger.debug("发现正在排队的任务{}，开始调度".format(tasks))

                    # 避免在目标服务器放辅助程序，使用ssh登录到目标服务器，获取服务器参数
                    # 注意：获取目标服务器信息的方式为，通过目标服务器上运行的glances服务获取
                    #       要求目标服务器必须安装 glances，并启动 glance web 服务（systemd）
                    mechines: List[MechineInfo] = []
                    for idx, ip in enumerate(_GPU_MECHINE_IPS):
                        timeout = aiohttp.ClientTimeout(total=5)
                        async with aiohttp.ClientSession(timeout=timeout) as session:
                            try:
                                async with session.get(
                                    "http://{}:61208/api/3/cpu".format(ip)
                                ) as resp:
                                    cpu = resp.json()["total"]
                                async with session.get(
                                    "http://{}:61208/api/3/mem".format(ip)
                                ) as resp:
                                    mem = resp.json()["percent"]
                                async with session.get(
                                    "http://{}:61208/api/3/gpu".format(ip)
                                ) as resp:
                                    gpu_info = resp.json()[0]
                                    gpu = gpu_info[
                                        "proc"
                                    ]  # FIXME: GPU应该枚举，此处只认为有一个gpu卡
                                    gpu_mem = gpu_info["mem"]
                                mechines.append(
                                    MechineInfo(
                                        id=idx,
                                        cpu=cpu,
                                        mem=mem,
                                        gpu=gpu,
                                        gpu_mem=gpu_mem,
                                    )
                                )
                            except:
                                logger.error("获取GPU服务器状态信息过程出错")

                    logger.debug("获取GPU服务器状态信息：{}".format(mechines))

                    result = schedule_task(mechines, tasks)
                    logger.debug("完成一次调度，结果为：{}".format(result))
                    for mechine_id, task_internal_info in result:
                        await start_task(mechine_id, task_internal_info)
                        await rds.srem("queued_tasks", task_internal_info.id)
                        await rds.sadd("running_tasks", task_internal_info.id)
                        await rds.hmset_dict(
                            task_internal_info.id, {"state": "running"}
                        )
        finally:
            rds.close()
            await rds.wait_closed()


# 放入调度任务
#asyncio.create_task(task_scheduler())


@router.get("/tasks")
async def get_task_list(
    current_user: JWTUser = Depends(get_current_user),
    redis: Redis = Depends(get_redis_connection),
) -> List[Task]:
    rst: List[Task] = []
    container_ids = await redis.smembers(current_user.id)
    for id in container_ids:
        v = await redis.hgetall(id, encoding="utf-8")
        if v["state"] == "running":
            s = "正在执行"
        elif v["state"] == "queue":
            s = "正在排队"
        else:
            s = "已停止"
        rst.append(Task(id=id, name=v["name"], state=s, mechine_id=v["mechine_id"]))
    return rst


@router.post("/add")
async def add_task(
    task_info: AddTaskInfo,
    current_user: JWTUser = Depends(get_current_user),
    redis: Redis = Depends(get_redis_connection),
):
    # 注意：添加函数并不创建容器，只有再调度后才创建容器
    # task id 由uuid库随机生成
    task_id = uuid.uuid1().hex
    await redis.sadd(current_user.id, task_id)
    await redis.hmset_dict(
        task_id,
        {
            "id": task_id,
            "name": task_info.name,
            "image": task_info.image,
            "container_id": "-1",  # 创建之初没有容器id，只有容器创建后才有id
            "mechine_id": -1,
            "btime": time.time(),
            "rtime": 0,
            "state": "queue",  # 初始化在列队状态
        },
    )
    await redis.sadd("queued_tasks", task_id)  # 存入全局任务列队

    return task_id


@router.post("/do")
async def do_some_actions(
    action: Action,
    current_user: JWTUser = Depends(get_current_user),
    redis: Redis = Depends(get_redis_connection),
):
    # 必须先验证用户提交的 taskid 是否归属自己
    if not await redis.sismember(current_user.id, action.taskid):
        raise HTTPException(HTTP_400_BAD_REQUEST)

    mechine_id = int(await redis.hget(action.taskid, "mechine_id", encoding="utf-8"))

    if action.action == "stop":
        if mechine_id >= 0:  # 只有开始运行，才能执行停止
            container_id = await redis.hget(
                action.taskid, "container_id", encoding="utf-8"
            )
            docker = aiodocker.Docker(GPU_MECHINES[mechine_id])
            container = docker.containers.container(container_id)
            await container.stop()
            await docker.close()
            # 任务停止后，需要从运行列队中删除，设置状态
            await redis.srem("running_tasks", action.taskid)
            await redis.hmset_dict(action.taskid, {"state": "stop"})
        else:
            raise HTTPException(HTTP_400_BAD_REQUEST)

    elif action.action == "delete":
        if mechine_id >= 0:  # 开始运行后，需要删除容器，删除redis中任务相关信息
            container_id = await redis.hget(
                action.taskid, "container_id", encoding="utf-8"
            )
            docker = aiodocker.Docker(GPU_MECHINES[mechine_id])
            container = docker.containers.container(container_id)
            await container.delete(force=True)
            await docker.close()
            await redis.srem("running_tasks", action.taskid)  # 任务停止后，需要从运行列队中删除
        else:
            await redis.srem("queued_tasks", action.taskid)  # 如果任务还没有被调度，需要从等待调度列队中删除

        await redis.delete(action.taskid)
        await redis.srem(current_user.id, action.taskid)


@router.websocket("/{task_id}/log")
async def read_log(
    task_id: str,
    ws: WebSocket,
    current_user: JWTUser = Depends(get_current_user),
    redis: Redis = Depends(get_redis_connection),
):
    await ws.accept()

    # 必须先验证用户提交的taskid是否归属自己
    if not await redis.sismember(current_user.id, task_id):
        await ws.close(code=WS_1008_POLICY_VIOLATION)
        return

    # 验证是否处在列队状态，除此状态之外都可以获取日志信息
    state = await redis.hget(task_id, "state", encoding="utf-8")
    if state == "queue":
        await ws.close(code=WS_1008_POLICY_VIOLATION)
        return

    try:
        task_internal_info: TaskInternal = TaskInternal(
            **await redis.hgetall(task_id, encoding="utf-8")
        )
        docker = aiodocker.Docker(GPU_MECHINES[int(task_internal_info.mechine_id)])
        container = docker.containers.container(task_internal_info.container_id)
        for d in container.log(stdout=True, stderr=True, follow=True):
            print(type(d), d)
            # await ws.send_bytes(d)
    except Exception as e:
        print(e)
        ws.close(code=WS_1008_POLICY_VIOLATION)

