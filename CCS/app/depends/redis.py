import aioredis
from aioredis import Redis
from app.configuration import REDIS_HOST


async def get_redis_connection() -> Redis:
    redis = await aioredis.create_redis_pool((REDIS_HOST, 6379))
    try:
        yield redis
    finally:
        redis.close()
        await redis.wait_closed()

async def get_redis_connection_manu()->Redis:
    redis = await aioredis.create_redis_pool((REDIS_HOST, 6379))
    return redis

