from pymongo import MongoClient
from app.configuration import MONGODB_HOST


async def get_mongodb_connection() -> MongoClient:
    mongoclient:MongoClient = MongoClient(host=MONGODB_HOST, port=27017)
    try:
        yield mongoclient
    finally:
        mongoclient.close()

