FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

LABEL maintainer="WANG Guan <netspecters@gmail.com>"

WORKDIR /app/

COPY requirements.txt /app/
RUN sed -i 's/deb.debian.org/mirrors.ustc.edu.cn/g' /etc/apt/sources.list \
    && apt update  \
    && apt install -y build-essential \
    && pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt

COPY ./app /app/app