from typing import List, Optional, Tuple, Union, cast
from pymongo import MongoClient
from pydantic import BaseModel
from fastapi import APIRouter, Depends, Form, File, UploadFile
from fastapi.exceptions import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
from app.configuration import MONGODB_HOST
from app.depends.jwt_auth import get_current_user, JWTUser
from app.depends.redis import get_redis_connection, get_redis_connection_manu
from app.depends.mongodb import get_mongodb_connection
from app.persistence.workflow import wfs_mongo
from aioredis import Redis
import time
from bson import json_util
from app.utils.wfs_utils import workflowUtil, imageUtil, workflowUtilFactory, grpcUtil
from bson.objectid import ObjectId

user_prefix = 'user_'
workflow_prefix = 'workflow_'

router = APIRouter()


class WorkflowTask(BaseModel):
    id: str = None
    name: str = None
    template: str = None
    phase: str = None
    dependencies: List[str] = []


class WorkflowTemplate(BaseModel):
    workflow_name: str
    topology: List[WorkflowTask] = None


@router.post('/workflows/{workflow_ID}/execute')
async def exec_workflow(workflow_ID: str = None,
                        redis: Redis = Depends(get_redis_connection),
                        mongodb_client: MongoClient = Depends(get_mongodb_connection),
                        current_user: JWTUser = Depends(get_current_user)):
    # 根据提交模版ID执行工作流，获取工作流信息
    workflow = wfs_mongo.find_workflow(mongodb_client, workflow_ID)

    workflow_name, creation_timestamp = workflowUtilFactory.instance.get_workflow_util()\
        .submit_workflow(workflow['yaml'])
    await redis.sadd(user_prefix + str(ObjectId(current_user.id)), workflow_name)
    workflow_record = {
        'workflow_name': workflow_name,
        'template_id': workflow_ID,
        'creation_timestamp': creation_timestamp
    }
    await redis.hmset_dict(workflow_prefix + workflow_name, workflow_record)
    grpcUtil.package(workflow['name'],workflow['topology'])

    return workflow_name


@router.post('/workflows')
async def upload_workflow(
        workflow: WorkflowTemplate = None,
        mongodb_client: MongoClient = Depends(get_mongodb_connection),
        current_user: JWTUser = Depends(get_current_user)):
    if None == workflow.workflow_name or '' == workflow.workflow_name or workflow.topology == None:
        return {'message': 'fail'}
    response = imageUtil.instance.submit_workflow(workflow.workflow_name, workflow.topology)
    task_list_temp = []
    for task in workflow.topology:
        task_list_temp.append(dict(task))
    workflow_response = {'yaml': response, 'name': workflow.workflow_name, 'topology': task_list_temp,
                         'create_time': time.time()}
    print(workflow_response)
    res = wfs_mongo.insert_data(mongodb_client, workflow_response)
    wfs_mongo.create_wf_to_user(mongodb_client, ObjectId(current_user.id), str(res))
    return json_util.dumps(workflow_response)


@router.patch('/workflows/{workflow_ID}')
async def update_workflow(
        workflow_ID: str = '', workflow_body: WorkflowTemplate = None,
        mongodb_client: MongoClient = Depends(get_mongodb_connection),
        current_user: JWTUser = Depends(get_current_user)):
    if None == workflow_ID or '' == workflow_ID or workflow_body.topology == None:
        return {'message': 'fail'}
    workflow = wfs_mongo.find_workflow(mongodb_client, workflow_ID)
    response = imageUtil.instance.submit_workflow(workflow_body.workflow_name, workflow_body.topology)
    task_list_temp = []
    for task in workflow_body.topology:
        task_list_temp.append(dict(task))
    workflow_response = {'yaml': response, 'name': workflow_body.workflow_name, 'topology': task_list_temp,
                         'create_time': time.time()}
    # print(workflow_response)
    res = wfs_mongo.update_data(mongodb_client, ObjectId(workflow_ID), workflow_response)
    return "success"


# ==============workflow job function

@router.get('/workflowJobs/{workflowjob_name}')
async def check_workflowjob_status(
        workflowjob_name: str = '',
        redis: Redis = Depends(get_redis_connection),
        current_user: JWTUser = Depends(get_current_user)):
    workflow = workflowUtilFactory.instance.get_workflow_util().get_workflow_status(workflowjob_name)
    return json_util.dumps(workflow)


@router.get('/workflowJobs')
async def get_workflow_job_list(redis: Redis = Depends(get_redis_connection),
                                current_user: JWTUser = Depends(get_current_user)):
    status_map = workflowUtilFactory.instance.get_workflow_util().list_namespaced_custom_object()
    workflow_list = await redis.smembers(user_prefix + current_user.id, encoding='utf-8')
    response = []
    for workflow in workflow_list:
        detail = await redis.hgetall(workflow_prefix + str(workflow), encoding='utf-8')
        detail['phase'] = status_map[detail['workflow_name']]
        # del detail['template_id']
        response.append(detail)
    return json_util.dumps(response)


@router.delete('/workflowJobs/{workflow_name}')
async def delete_workflowjob(workflow_name: str, redis: Redis = Depends(get_redis_connection),
                             current_user: JWTUser = Depends(get_current_user)):
    workflowUtilFactory.instance.get_workflow_util().delete_namespaced_custom_object(workflow_name)
    await redis.srem(user_prefix + current_user.id, workflow_name)
    await redis.delete(workflow_prefix + workflow_name)
    return json_util.dumps({'message', 'success'})


@router.patch('/workflowJobs/{workflowjob_name}/suspend')
async def suspend_job(workflowjob_name: str, redis: Redis = Depends(get_redis_connection),
                      current_user: JWTUser = Depends(get_current_user)):
    workflowUtilFactory.instance.get_workflow_util().patch_namespaced_custom_object(workflowjob_name, True)
    await redis.hset(workflow_prefix + workflowjob_name, 'status', 'Suspended')
    return json_util.dumps({'message', 'success'})


# 对工作流的增删改查
@router.get("/workflow_list/WFFInfo")
async def SearchWFFInfo(current_user: JWTUser = Depends(get_current_user),
                        mongodb_client: MongoClient = Depends(get_mongodb_connection)):
    """
    查询用户自己的工作流
    """
    db = mongodb_client["ccs"]
    wf_collection = db["workflow"]
    user_collection = db['users']

    wf_dict = []
    try:
        db_user_wf = user_collection.find_one({"_id": ObjectId(current_user.id)}, {"workflows": 1})
        if db_user_wf is None:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="当前用户不存在！")
        else:
            db_user_wf = db_user_wf['workflows']
            wf_info = []
            for item in db_user_wf:
                wf_info = wf_collection.find_one({'_id': ObjectId(item)}, {"_id": 1, "create_time": 1, "name": 1})

                mydict = {}
                mydict["wf_id"] = format(wf_info['_id']).__str__()

                mydict["date"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(wf_info['create_time']))
                mydict["name"] = format(wf_info['name'])
                wf_dict.append(mydict)
            return (wf_dict)
    except HTTPException:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="查询过程出现错误！")


@router.get("/workflow_list/WFTInfo")
async def SearchWFTInfo(current_user: JWTUser = Depends(get_current_user),
                        mongodb_client: MongoClient = Depends(get_mongodb_connection)):
    """
    查询分享给自己的工作流
    """
    db = mongodb_client["ccs"]
    wf_collection = db["workflow"]
    share_collection = db['sharing_relationship']

    to_wf_dict = []
    try:
        db_to_wf = share_collection.find({"to_user_id": current_user.id}, {"wf_id": 1})
        result_to_wf_list = []  # 所有需要查找的wf id
        for ele in db_to_wf:
            id = ele['wf_id']
            result_to_wf_list.append(id)
        for item in result_to_wf_list:
            temp_info = wf_collection.find_one({'_id': ObjectId(item)}, {"_id": 1, "create_time": 1, "name": 1})
            mydict = {}
            mydict["wf_id"] = format(temp_info['_id']).__str__()
            # mydict["date"] = format(temp_info['create_time'])
            mydict["date"] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(temp_info['create_time']))
            mydict["name"] = format(temp_info['name'])
            to_wf_dict.append(mydict)
        return to_wf_dict
    except HTTPException:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="查询过程出现错误！")


@router.delete("/workflow_list/WFTInfo/{wf_id}")
async def cancel_share(wf_id, current_user: JWTUser = Depends(get_current_user),
                       mongodb_client: MongoClient = Depends(get_mongodb_connection)):
    """
    取消该用户 被分享的 工作流
    :param item:
    :param current_user:
    :param mongodb_client:
    :return:
    """
    if wf_id is None:
        return "fail"
    else:
        db = mongodb_client['ccs']
        col = db['sharing_relationship']
        try:
            res = col.delete_one({"wf_id": wf_id, "to_user_id": current_user.id})
            return "success"
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="取消分享过程中出现错误，详细信息参见后端日志"
            )


@router.get("/workflow_list/{workflow_id}")
async def workflow_details(workflow_id: str, mongodb_client: MongoClient = Depends(get_mongodb_connection)):
    """
    根据workflow id 获取网页 工作流的拓扑结构
    :param workflow_id:
    :param mongodb_client:
    :return:
    """
    mydb = mongodb_client["ccs"]
    workflow_col = mydb['workflow']

    res = workflow_col.find_one({"_id": ObjectId(workflow_id)})
    if res is None:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="该工作流不存在！")
    topology = res["topology"]
    mydict = []
    for ele_node in topology:
        temp = {}
        temp['name'] = ele_node['name']
        temp['dependencies'] = ele_node['dependencies']
        temp['id'] = ele_node['id']
        temp['template'] = ele_node['template']
        temp['phase'] = 'normal'
        mydict.append(temp)
    return mydict


@router.post("/workflow_list/{wf_id}")
async def copy_workflow(wf_id, current_user: JWTUser = Depends(get_current_user),
                        mongodb_client: MongoClient = Depends(get_mongodb_connection)):
    """
    复制工作流
    :param item:
    :param current_user:
    :param mongodb_client:
    :return:
    """
    if wf_id is None:
        return "fail"

    db = mongodb_client['ccs']
    wf_col = db['workflow']
    user_col = db['users']

    copy_temp_data = wf_col.find_one({"_id": ObjectId(wf_id)},{'_id':0})
        
    copy_insert_data = copy_temp_data
    copy_insert_data['name'] = copy_temp_data['name'] + "-副本"
    copy_insert_data['create_time'] = time.time()

    try:
        wf_res = wf_col.insert_one(copy_insert_data).inserted_id
        wf_id = str(wf_res)
        user_res = user_col.update_one({'_id': ObjectId(current_user.id)}, {"$push": {"workflows": wf_id}})
        if user_res is None:
            return "fail"
        else:
            return "success"
    except Exception:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="复制过程中出现错误，详细信息参见后端日志"
        )


@router.delete("/workflow_list/WFFInfo/{wf_id}")
async def delete_workflow(wf_id, current_user: JWTUser = Depends(get_current_user),
                          mongodb_client: MongoClient = Depends(get_mongodb_connection)):
    """
    删除用户自己创建的工作流
    :param item:
    :param current_user:
    :param mongodb_client:
    :return:
    """

    if wf_id is None:
        return "fail"
    else:
        db = mongodb_client['ccs']
        user_col = db['users']
        wf_col = db['workflow']
        try:
            res = user_col.update_one({'_id': ObjectId(current_user.id)}, {"$pull": {"workflows": wf_id}})
            res = wf_col.delete_one({'_id': ObjectId(wf_id)})
            return "success"

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="复制过程中出现错误，详细信息参见后端日志"
            )


@router.get("/workflow_list/{wf_id}/share")
async def SearchShareInfo(wf_id, current_user: JWTUser = Depends(get_current_user),
                          mongodb_client: MongoClient = Depends(get_mongodb_connection)):
    """
    查询用户分享信息
    """
    db = mongodb_client["ccs"]
    share_collection = db["sharing_relationship"]
    to_user_collection = db['users']

    wfid = wf_id
    user_dict = []

    try:
        db_to_user = share_collection.find({"wf_id": wfid, "from_user_id": current_user.id},
                                           {"to_user_id": 1})
        for ele in db_to_user:
            db_to_user_id = ele['to_user_id']
            to_user_name = to_user_collection.find_one({'_id': ObjectId(db_to_user_id)}, {"_id": 1, "email": 1})
            mydict = {}
            mydict["to_user_id"] = format(to_user_name['_id']).__str__()
            mydict["name"] = format(to_user_name['email'])
            user_dict.append(mydict)
        return user_dict
    except HTTPException:
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="添加过程中出现错误，详细信息参见后端日志"
        )


@router.post("/workflow_list/{wf_id}/share/{to_user_name}")
async def add_share(to_user_name: str, wf_id: str, current_user: JWTUser = Depends(get_current_user),
                    mongodb_client: MongoClient = Depends(get_mongodb_connection)):
    """
    当前用户分享工作流给其他新用户
    :param item:
    :param current_user:
    :param mongodb_client:
    :return:
    """
    db = mongodb_client['ccs']
    share_collection = db['sharing_relationship']

    user_collection = db['users']

    # 根据email拿到用户的信息,用户具有唯一性
    to_user_data = user_collection.find_one({"email": to_user_name})
    if to_user_data is None:
        return "fail"
    to_user_id = str(to_user_data['_id'])
    res = share_collection.find_one(
        {"wf_id": wf_id, "from_user_id": current_user.id, "to_user_id": to_user_id})

    if res is None:
        #  是否插入成功
        try:
            insert_data = {"wf_id": wf_id, "from_user_id": current_user.id, "to_user_id": to_user_id}
            res = share_collection.insert_one(insert_data)
            return "success"
        except Exception:
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="添加过程中出现错误，详细信息参见后端日志"
            )
    else:  # 该用户已经被分享
        return "fail"


@router.delete('/workflow_list/{wf_id}/share/{to_user_name}')
async def share_delete(wf_id, to_user_name, current_user: JWTUser = Depends(get_current_user),
                       mongodb_client: MongoClient = Depends(get_mongodb_connection)):
    """
    用户 点开工作流 删除该工作流被分享的用户
    :param item:
    :param current_user:
    :param mongodb_client:
    :return:
    """
    db = mongodb_client['ccs']
    col = db['sharing_relationship']
    user_collection = db['users']

    to_user_data = user_collection.find_one({'email': to_user_name})
    if to_user_data is None:
        return "fail"
    else:
        to_user_id = str(to_user_data['_id'])
        try:
            delete_data = {"wf_id": wf_id, "from_user_id": current_user.id, "to_user_id": to_user_id}
            res = col.delete_one(delete_data)
            return "success"
        except Exception:
            import traceback
            traceback.print_exc()

            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="删除过程中出现错误，详细信息参见后端日志"
            )


# class UpdateWFSInfo(BaseModel):
#     wfs_name: str
#     topology: list
#
#
# @router.post("/workflow_list/add")
# async def create_new_wf(item:UpdateWFSInfo,current_user: JWTUser = Depends(get_current_user), mongodb_client: MongoClient = Depends(get_mongodb_connection)):
#     mydb = mongodb_client['ccs']
#     mycol_workflow = mydb['workflow']
#     my_user = mydb['users']
#     try:
#         # res = mycol_workflow.update_one({'_id': ObjectId(workflow_id)},
#         #                                 {"$set": {"topology": item.topology, "name": item.wfs_name}})
#         wf_id = mycol_workflow.insert_one({'topology':item.topology,'name':item.wfs_name,'create_time':str(datetime.datetime.now())}).inserted_id
#         res = my_user.update_one({'_id':ObjectId(current_user.id)},{"$push": {"workflows": str(wf_id)}})
#         return "success"
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(
#             status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="新增过程中出现错误，详细信息参见后端日志"
#         )
#
# @router.patch("/workflow_list/{workflow_id}/update/")
# async def update_wfs_info(workflow_id, item: UpdateWFSInfo,
#                           mongodb_client: MongoClient = Depends(get_mongodb_connection)):
#     mydb = mongodb_client['ccs']
#     mycol_workflow = mydb['workflow']
#
#     print("................")
#     print(workflow_id)
#     print(item.topology)
#
#     try:
#         res = mycol_workflow.update_one({'_id': ObjectId(workflow_id)},
#                                         {"$set": {"topology": item.topology, "name": item.wfs_name}})
#         return "success"
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(
#             status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="修改过程中出现错误，详细信息参见后端日志"
#         )


@router.patch('/workflowJobs/{workflowjob_name}/resume')
async def suspend_job(workflowjob_name: str, redis: Redis = Depends(get_redis_connection)):
    workflowUtilFactory.instance.get_workflow_util().patch_namespaced_custom_object(workflowjob_name, False)
    await redis.hset(workflow_prefix + workflowjob_name, 'status', 'Running')
    return json_util.dumps({'message', 'success'})
