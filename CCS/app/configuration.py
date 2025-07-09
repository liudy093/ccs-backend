# 默认值
LOGGING_NAME="ccs.backend"
LOG_LEVEL = "DEBUG" # NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL

JWT_SECRET = "BZz4bHabcQD?g9YN2aksBn7*r3P(eo]P,Dt8NCEKs6VP34qmTL#8f&ruD^TtG"
JWT_ALGORITHM = "HS256"
JWT_USER_ID_FIELD_NAME = "id"  # jwt token 中 user id 字段的名称
JWT_USER_NAME_FIELD_NAME = "username"
JWT_USER_EMAIL_FIELD_NAME = "email"

MONGODB_HOST = "localhost"
REDIS_HOST = "localhost"
INFLUXDB_HOST="localhost"
GPU_MECHINE_IPS = "10.1.80.79,10.1.80.78"


# ---------------- 环境获取 --------------------


# 尝试从环境中获取
# 环境中没有相关值，就用默认值
import os

LOG_LEVEL = os.environ.get("LOG_LEVEL", LOG_LEVEL)

JWT_SECRET = os.environ.get("JWT_SECRET", JWT_SECRET)
JWT_ALGORITHM = os.environ.get("JWT_ALGORITHM", JWT_ALGORITHM)
JWT_USER_ID_FIELD_NAME = os.environ.get(
    "JWT_USER_ID_FIELD_NAME", JWT_USER_ID_FIELD_NAME
)
JWT_USER_NAME_FIELD_NAME = os.environ.get(
    "JWT_USER_NAME_FIELD_NAME", JWT_USER_NAME_FIELD_NAME
)
JWT_USER_EMAIL_FIELD_NAME = os.environ.get(
    "JWT_USER_EMAIL_FIELD_NAME", JWT_USER_EMAIL_FIELD_NAME
)
MONGODB_HOST = os.environ.get("MONGODB_HOST", MONGODB_HOST)
INFLUXDB_HOST = os.environ.get("INFLUXDB_HOST ", INFLUXDB_HOST)
REDIS_HOST = os.environ.get("REDIS_HOST", REDIS_HOST)
GPU_MECHINE_IPS = os.environ.get("GPU_MECHINE_IPS", GPU_MECHINE_IPS)
