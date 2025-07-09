from influxdb import InfluxDBClient
from app.configuration import INFLUXDB_HOST


async def get_influxdb_connection() -> InfluxDBClient:
    influxclient:InfluxDBClient = InfluxDBClient("localhost", 8086, '', '', 'medical')
    try:
        yield influxclient
    finally:
        influxclient.close()

