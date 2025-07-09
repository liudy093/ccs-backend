"""
基于Redis构建的分布式锁
"""
import asyncio
import time

import aioredis

# time unit all is milliseconds
DEFAULT_RETRY_TIMES = 1  # 重试1次
DEFAULT_RETRY_INTERVAL = 50  # 重试等待 50ms
DEFAULT_EXPIRE = 30000  # 默认锁超时 30s


class RedLock:
    def __init__(
        self,
        rds: aioredis.Redis,
        key,
        expire=DEFAULT_EXPIRE,
        retry_times=DEFAULT_RETRY_TIMES,
        retry_interval=DEFAULT_RETRY_INTERVAL,
    ):
        """
        Ref: https://redis.io/topics/distlock

        :param rds: redis connection
        :param key: lock key
        :param expire:  lock key expire ttl(ms)
        :param retry_times: retry times to get lock
        :param retry_interval: retry interval(ms)

        total max time is: retry_times * retry_interval(ms)
        """
        if not isinstance(rds, aioredis.Redis):
            raise Exception("rds should be an instance of aioredis.Redis")
        self.rds: aioredis.Redis = rds
        self.key = key
        self.expire = expire
        self.retry_times = retry_times
        self.retry_interval = retry_interval
        self.value = time.time() * 1000 + self.expire  # ms

    async def __aenter__(self):
        return await self.lock()

    async def __aexit__(self, exc_type, exc_value, traceback):
        await self.release()

    async def release(self):
        if time.time() * 1000 < self.value:
            await self.rds.delete(self.key)
        else:
            _value = await self.rds.get(self.key)
            if _value == self.value:
                await self.rds.delete(self.key)
            raise Exception("handle timeout after got redlock")

    async def lock(self):
        for retry in range(self.retry_times + 1):
            if await self.rds.set(
                self.key, self.value, pexpire=self.expire, exist="SET_IF_NOT_EXIST"
            ):
                return True
            await asyncio.sleep(self.retry_interval / 1000)

        return False
