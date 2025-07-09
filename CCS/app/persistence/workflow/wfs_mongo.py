from typing import List, Optional, Tuple, Union, cast
from pymongo import MongoClient
from bson.objectid import ObjectId
from pydantic import BaseModel
from fastapi import APIRouter, Depends
from fastapi.exceptions import HTTPException
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
from app.configuration import MONGODB_HOST
from app.depends.jwt_auth import get_current_user, JWTUser

router = APIRouter()


# 1.插入数据
def insert_data(client, data):
    if data is None:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail='插入的数据错误')
    else:
        mydb = client['ccs']
        mycol = mydb['workflow']
        if type(data) == dict:
            res = mycol.insert_one(data).inserted_id
        else:
            res = mycol.insert_many(data).inserted_ids
        return str(res)


# 1.更新数据
def update_data(client, id, data):
    if data is None:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail='更新的数据错误')
    else:
        mydb = client['ccs']
        mycol = mydb['workflow']
        query = { '_id': id}
        res = mycol.update_one(query, { '$set': data})
        return res


# 2.查询,给定一个name和namespace，查找指定的一个工作流信息，全部返回
def find_workflow(client, id: str):
    id = ObjectId(id)
    mydb = client['ccs']
    mycol = mydb['workflow']
    try:
        print(id)
        mydoc = mycol.find_one({'_id': ObjectId(id)})
        if mydoc is None:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail='该工作流不存在！')
        else:
            return mydoc
    except Exception:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail='查询过程中出现错误，详细信息参见后端日志'
        )


# 返回工作流id列表
def find_workflow_ids(client):
    mydb = client['ccs']
    mycol = mydb['workflow']

    try:
        res = mycol.find({}, {'_id': 1})
        if res is None:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail='该工作流不存在！')
        else:
            return res
    except Exception:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail='查询过程中出现错误，详细信息参见后端日志'
        )


# 删除信息
class DEL(BaseModel):
    id: str = None
    id1: str = None
    id2: str = None
    id3: str = None
    id4: str = None


async def delete_user(client, item: DEL, current_user: JWTUser = Depends(get_current_user)):
    '''
    根据用户名删除该用户下的所有工作流信息
    :param item:
    :return:
    '''
    if item is None:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail='插入的数据错误')
    else:
        item_dict = item.dict()
        mydb = client['ccs']
        mycol = mydb['workflow']
        for value in item_dict.values():
            mycol.delete_one({'userid': value})
        return current_user.id

def create_wf_to_user(client,user_id,wf_id):
    my_db = client['ccs']
    my_user = my_db['users']
    res = my_user.update_one({'_id': user_id}, {"$push": {"workflows": wf_id}})


def delete_workflow(client, item: DEL, namespace='default', current_user: JWTUser = Depends(get_current_user)):
    '''
    用户删除自己的工作流信息,del中变量为工作流name
    :param item:
    :param current_user:
    :return:
    '''
    if item is None:
        raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail='插入的数据错误')
    else:
        item_dict = item.dict()
        mydb = client['ccs']
        mycol = mydb['workflow']
        for value in item_dict.values():
            mycol.delete_many({"metadata.name": value, "metadata.namespace": namespace})
        return "delete complete!"


if __name__ == "__main__":
    mclient = MongoClient(host="localhost", port=27017)
    mydb = mclient["ccs"]
    mycol_users = mydb["users"]
    mycol_workflow = mydb["workflow"]
    share_collection = mydb['sharing_relationship']

    # res = mycol_workflow.find_one({'_id': ObjectId('5dfcf20a434ff41072572a3c')})
    # print(res)

    '''
    使用user_id做测试，上线时更替成current_user.id
    '''
    user_id = '5df204ad48cec4383cdf62d0'
    wf_id = "5dfb571b18e1591f50e6b5ba"
    res = mycol_users.insert_one({"username":"bai","password":"123","email":"111@163.com","nickname":"fei","role":"1","workflows":[]})
    # rr = mycol_users.find()
    # for ob in rr:
    #     print(ob)
    # rrr = mycol_users.find_one({'username':'bai'})
    # print('********')
    # print(rrr)
    # res = share_collection.find({'to_user_id':user_id})


    # for ele in res:
    #     print(ele['_id'])
    #     print(type(ele['_id']))
    #     if str(ele['_id']) == '5dfcf20a434ff41072572a3c':
    #         print(ele)
    #         for t in ele:
    #             print(t + ':' + ele[t])

