import os
from datetime import datetime
from typing import Any, List, Union
from fastapi import FastAPI

from .configuration import JWT_SECRET, LOGGING_NAME, LOG_LEVEL
from .controller import users, medical, gp, wfs
import logging
from .controller.medical import workerinput
from .controller.medical import workerinput_svm_knn
import asyncio

app = FastAPI(
    title="Backend of Cloud Control Systems",
    description="云控制一体化平台——后端 RESTFul Interface",
    version="0.1.2",
    redoc_url=None,
)


@app.get("/test")
def just_hello_for_test():
    return {"hello": "world"}

# asyncio.ensure_future(worker())
asyncio.ensure_future(workerinput())
asyncio.ensure_future(workerinput_svm_knn())

app.include_router(users.router, prefix="/api/users")
app.include_router(medical.router, prefix="/api/medical")
app.include_router(wfs.router, prefix="/api/wfs", tags=["Workflow Scheduler System (wfs)"])
app.include_router(gp.router, prefix="/api/gp", tags=["General Purpose Compute (GP)"])

# 构建日志系统
logger = logging.getLogger(LOGGING_NAME)  # 所有子系统的日志均需要从此处开始扩展
logger.setLevel(LOG_LEVEL)
logger.info("后端已启动")
