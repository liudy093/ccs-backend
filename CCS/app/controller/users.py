from typing import List, Optional, Tuple, Union, cast
from pymongo import MongoClient
from pydantic import BaseModel
from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
from app.depends.jwt_auth import encode_token
from app.configuration import MONGODB_HOST
from app.depends.mongodb import get_mongodb_connection

router = APIRouter()


class LoginResult(BaseModel):
    token_type: str
    access_token: str
    stuid: str
    username: str
    role: str
    medicalsignup: str


@router.post("/login", response_model=LoginResult)
def login(
    login_info: OAuth2PasswordRequestForm = Depends(),
    mongodb_client: MongoClient = Depends(get_mongodb_connection),
) -> LoginResult:
    """
    用户登录
    """
    username = login_info.username
    password = login_info.password
    db = mongodb_client["ccs"]
    user_collection = db["users"]

    try:
        db_user = user_collection.find_one({"username": username, "password": password})
        if db_user is None:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="用户名或密码错误！")
        return LoginResult(
            token_type="bearer",
            access_token=encode_token(
                {
                    "id": str(db_user["_id"]),
                    "username": db_user["username"],
                    "email": db_user["email"],
                    "role": db_user["role"],
                }
            ),
            username=db_user["username"],
            stuid=db_user["stuid"],
            role=db_user["role"],
            medicalsignup=db_user["medicalsignup"],
        )

    except HTTPException:
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="登录过程中出现错误，详细信息参见后端日志"
        )


##注册
class Signup(BaseModel):
    email: str
    password: str
    repassword: str
    nickname: str
    username: str
    phone: str
    stuid: str
    idcard: str


@router.post("/signup")
def signup(
    siginup_info: Signup, mongodb_client: MongoClient = Depends(get_mongodb_connection)
):
    """
    用户注册
    """
    email = siginup_info.email
    password = siginup_info.password
    nickname = siginup_info.nickname
    role = "user"
    username = siginup_info.username
    phone = siginup_info.phone
    stuid = siginup_info.stuid
    idcard = siginup_info.idcard
    workflows = []

    db = mongodb_client["ccs"]
    user_collection = db["users"]

    try:
        result = user_collection.find_one()
        if result is None:
            role = "admin"
        db_user = user_collection.find_one({"username": username})
        if db_user is None:
            res = user_collection.insert_one(
                {
                    "email": email,
                    "password": password,
                    "nickname": nickname,
                    "username": username,
                    "role": role,
                    "phone": phone,
                    "stuid": stuid,
                    "idcard": idcard,
                    "workflows": workflows,
                    "medicalsignup":'N'
                }
            )
            return "success"
        else:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="用户已存在！")

    except HTTPException:
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="注册过程中出现错误，详细信息参见后端日志"
        )


##得到用户
@router.get("/getUserInfo")
async def getUserInfo(mongodb_client: MongoClient = Depends(get_mongodb_connection)):
    """
    用户信息表
    """
    db = mongodb_client["ccs"]
    user_collection = db["users"]
    user_dict = []
    try:
        db_user = user_collection.find()
        result_list = list(db_user[:])
        for document in result_list:
            mydict = {}
            mydict["email"] = format(document["email"])
            mydict["nickname"] = format(document["nickname"])
            mydict["role"] = format(document["role"])
            mydict["username"] = format(document["username"])
            mydict["phone"] = format(document["phone"])
            mydict["stuid"] = format(document["stuid"])
            mydict["idcard"] = format(document["idcard"])
            user_dict.append(mydict)
        return user_dict

    except HTTPException:
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="删除过程中出现错误，详细信息参见后端日志"
        )


##查询用户信息
class Searchuser(BaseModel):
    selectclass: str
    text: str


@router.post("/SearchUserInfo")
async def SearchUserInfo(
    searchuser_info: Searchuser,
    mongodb_client: MongoClient = Depends(get_mongodb_connection),
):
    """
    搜索用户
    """
    db = mongodb_client["ccs"]
    user_collection = db["users"]
    ukey = searchuser_info.selectclass
    uvalue = searchuser_info.text
    user_dict = []
    try:
        db_user = user_collection.find({ukey: uvalue})
        if db_user is None:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="没有这样的用户！")
        else:
            result_list = list(db_user[:])
            for document in result_list:
                mydict = {}
                mydict["email"] = format(document["email"])
                mydict["nickname"] = format(document["nickname"])
                mydict["role"] = format(document["role"])
                mydict["username"] = format(document["username"])
                mydict["phone"] = format(document["phone"])
                mydict["stuid"] = format(document["stuid"])
                mydict["idcard"] = format(document["idcard"])
                user_dict.append(mydict)
            return user_dict
    except HTTPException:
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="查询过程中出现错误，详细信息参见后端日志"
        )


##删除用户信息
class Deluser(BaseModel):
    email: str


@router.post("/delUserInfo")
async def delUserInfo(
    deluser_info: Deluser, mongodb_client: MongoClient = Depends(get_mongodb_connection)
):
    """
    删除用户
    """
    mclient = MongoClient(host=MONGODB_HOST, port=27017)
    db = mclient["ccs"]
    user_collection = db["users"]
    email = deluser_info.email
    print(email)
    try:
        res = user_collection.delete_one({"email": email})
        if res is None:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="没有这样的用户！")
        else:
            return "success"
    except HTTPException:
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="删除过程中出现错误，详细信息参见后端日志"
        )


##修改用户星星
class Updateuser(BaseModel):
    email: str
    nickname: str
    role: str
    username: str
    phone: str
    stuid: str
    idcard: str


@router.post("/updateUserInfo")
async def updateUserInfo(
    updateuser_info: Updateuser,
    mongodb_client: MongoClient = Depends(get_mongodb_connection),
):
    """
    修改用户
    """
    db = mongodb_client["ccs"]
    user_collection = db["users"]

    email = updateuser_info.email
    nickname = updateuser_info.nickname
    role = updateuser_info.role
    username = updateuser_info.username
    phone = updateuser_info.phone
    stuid = updateuser_info.stuid
    idcard = updateuser_info.idcard

    try:
        user_collection.update({"email": email}, {"$set": {"nickname": nickname}})
        user_collection.update({"email": email}, {"$set": {"role": role}})
        user_collection.update({"email": email}, {"$set": {"username": username}})
        user_collection.update({"email": email}, {"$set": {"phone": phone}})
        user_collection.update({"email": email}, {"$set": {"stuid": stuid}})
        user_collection.update({"email": email}, {"$set": {"idcard": idcard}})
        return "success"

    except HTTPException:
        raise

    except Exception as e:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="编辑过程中出现错误，详细信息参见后端日志"
        )
    print(id)
    return "success"

