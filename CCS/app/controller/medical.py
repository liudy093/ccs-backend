import re
from typing import List, Optional, Tuple, Union, cast
from numpy.core.fromnumeric import size
from numpy.core.records import array
from numpy.core.shape_base import _concatenate_shapes
from numpy.lib.function_base import append
from pymongo import MongoClient
from pydantic import BaseModel
from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.exceptions import HTTPException
from starlette.responses import Response
from starlette.status import HTTP_400_BAD_REQUEST, HTTP_500_INTERNAL_SERVER_ERROR
from app.configuration import MONGODB_HOST
from app.configuration import INFLUXDB_HOST
from app.depends.jwt_auth import get_current_user, JWTUser
from influxdb import InfluxDBClient
from app.depends.mongodb import get_mongodb_connection
from app.depends.influxdb import get_influxdb_connection
import datetime
#引入异步io和休眠函数
#注意，在写命令的时候，要避免数据库的重复读写，一次写很多遍的那种，这样会拖慢速度
import asyncio
import time
# 这两条是用来生成随机数的
import random
router = APIRouter()
######这是关于神经网络算法的部分####
import numpy as np
import _pickle as cp
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras import optimizers
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, TensorBoard
import json
from app.controller.tremor_detection_func import tremor_detection
from app.controller.feature_extraction_func import feature_extraction_func
from app.controller.outputfeature_func import outputfeature
from app.controller.func_classify2_off_line import outputfeature_svm_knn
#####这是关于神经网络算法的部分####


#worker函数要写到这里面，然后在main里面import进来
# queue = asyncio.Queue()
# async def worker():
#     ####部署神经网络算法
#     ##===========参数=================
#     log_dir = './app/controller/model/'
#     NB_SENSOR_CHANNELS = 1 #通道数目
#     NUM_CLASSES = 2       #类别数二分类
#     # BATCH_SIZE = 16
#     NUM_FILTERS = 64
#     FILTER_SIZE = 3
#     NUM_UNITS_LSTM = 64
#     STRIDE_SIZE=2 #卷积步长
#     time_step = 600
#     rmp = optimizers.RMSprop(lr=5e-4, rho=0.9, epsilon=1e-08, decay=1e-4)
#     model = Sequential()
#     model.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE, strides=STRIDE_SIZE,activation='relu',
#                  input_shape=(time_step, NB_SENSOR_CHANNELS), 
#                  name='Conv1D_1'))
#     model.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE,strides=STRIDE_SIZE, activation='relu', 
#                  name='Conv1D_2'))
#     model.add(Conv1D(filters=NUM_FILTERS, kernel_size=FILTER_SIZE,strides=STRIDE_SIZE, activation='relu', 
#                  name='Conv1D_3'))
#     model.add(LSTM(NUM_UNITS_LSTM, return_sequences=True, 
#                name='LSTM_1'))
#     model.add(LSTM(NUM_UNITS_LSTM, return_sequences=True, 
#                name='LSTM_2'))
#     model.add(Flatten(name='Flatten'))
#     model.add(Dropout(0.8, name='dropout'))
#     model.add(Dense(NUM_CLASSES, activation='softmax', 
#                 name='Output'))
#     model.compile(loss='poisson', \
#               optimizer=rmp, metrics=['accuracy'])
#     # print(model.summary())
#     checkpoint = ModelCheckpoint(log_dir + "best_weights.h5",
#                                  monitor="val_accuracy",
#                                  mode='max',
#                                  save_weights_only=True,
#                                  save_best_only=True, 
#                                  verbose=1,
#                                  period=1)
#     tensorboard = TensorBoard(log_dir=log_dir)
#     call_backs = [tensorboard, checkpoint]
#     model.load_weights('./app/controller/model/best_weights.h5')
#     # print(type(model))
#     ###部署神经网络算法
#     while True:
#         receive_data = await queue.get()
#         claresults=classifier(model,receive_data[0]) 
#         dt=receive_data[1]
#         # print(round(claresults,2))
#         client = InfluxDBClient("influxdb", 8086, '', '', 'medical')
#         body = [{
#          "measurement": "printresults",
#          "tags": {"id": receive_data[2]},
#          "time":dt,
#          "fields": {
#             "claresults": round(claresults,2)*100
#             },
#            }
#           ]
#         print(claresults)
#         client.write_points(body)
#         print("ok")



def classifier(model,receive_data):
    receive_data=np.array(receive_data)
    data_mean = np.mean(receive_data)
    data_std = np.std(receive_data)
    receive_data = (receive_data -data_mean)/data_std
    h2_data=receive_data.reshape(600,1)
    h2_data=np.array([h2_data])
    test_pred = np.argmax(model.predict(h2_data), axis=1)
    return model.predict(h2_data).tolist()[0][1]


#worker函数要写到这里面，然后在main里面import进来
queue1 = asyncio.Queue()
async def workerinput():
    samplingfrequency=20
    ###部署神经网络算法
    while True:
        receive_data = await queue1.get()
        client = InfluxDBClient("localhost", 8086, '', '', 'medical')
        contents1=receive_data[0]
        listofcontent=[]
        try:
            # print(contents1[0])
            size_of_hand=0
            size_of_foot=0
            listofcontent=[]
            i=0
            bingren1="bingren1"
            # print(contents1)
            # print(contents1)
            moving_detection=outputfeature(contents1)
            # print(moving_detection)
            while i<=(len(contents1)-40):
                if (contents1[i:i+4]=="AAAA") and (contents1[i+36:i+40]=="FFBB"):
                    listofcontent.append(contents1[i:i+40])
                    size_of_hand=size_of_hand+1
                    i=i+40
                elif (contents1[i:i+4]=="CCCC") and (contents1[i+36:i+40]=="FFBB"):
                    listofcontent.append(contents1[i:i+40])
                    size_of_foot=size_of_foot+1 
                    i=i+40
                else:
                    i=i+1 
            # print(listofcontent)
            # print(size_of_foot)
            # print("time_of_hand:"+str(int(size_of_hand*(1/samplingfrequency)/60)))
            # print("time_of_foot:"+str(int(size_of_foot*(1/samplingfrequency)/60)))
            intial_time_hand=nowh=receive_data[1]-datetime.timedelta(minutes=int(size_of_hand*(1/samplingfrequency)/60))
            nowf=receive_data[1]-datetime.timedelta(minutes=int(size_of_foot*(1/samplingfrequency)/60))
            # print("start_time_of_hand:"+str(nowh))
            # print("start_time_of_foot:"+str(nowf))
            body=[]
            rawdatalist=[[[] for _ in range(8)] for _ in range(2)]
            for index in range(0,len(listofcontent)):
                # print(listofcontent[index][8:20])
                acce_x=ComplementConv(int(listofcontent[index][4:6],16)|(int(listofcontent[index][6:8],16)<<8))*(8*9.86/32768)
                acce_y=ComplementConv(int(listofcontent[index][8:10],16)|(int(listofcontent[index][10:12],16)<<8))*(8*9.86/32768)
                acce_z=ComplementConv(int(listofcontent[index][12:14],16)|(int(listofcontent[index][14:16],16)<<8))*(8*9.86/32768)
                gyro_x=ComplementConv(int(listofcontent[index][16:18],16)|(int(listofcontent[index][18:20],16)<<8))*(2000/32768*(3.14/180))
                gyro_y=ComplementConv(int(listofcontent[index][20:22],16)|(int(listofcontent[index][22:24],16)<<8))*(2000/32768*(3.14/180))
                gyro_z=ComplementConv(int(listofcontent[index][24:26],16)|(int(listofcontent[index][26:28],16)<<8))*(2000/32768*(3.14/180))
                if listofcontent[index][0:4]=="AAAA":
                    nowh=nowh+datetime.timedelta(seconds=1/samplingfrequency)
                    dt=nowh
                    Identification="hand"
                    points ={
                        "measurement": Identification+"initial",
                        "tags": {"id":bingren1},
                        "time": dt,
                        "fields": {
                        "acce_x": acce_x,
                        "acce_y": acce_y,
                        "acce_z": acce_z,
                        "gyro_x": gyro_x,
                        "gyro_y": gyro_y,
                        "gyro_z": gyro_z,
                        "angle_x":ComplementConv(int(listofcontent[index][28:30],16)|(int(listofcontent[index][30:32],16)<<8))/100,
                        "angle_y":ComplementConv(int(listofcontent[index][32:34],16)|(int(listofcontent[index][34:36],16)<<8))/100
                        },
                    }
                    rawdatalist[0][0].append(acce_x)
                    rawdatalist[0][1].append(acce_y)
                    rawdatalist[0][2].append(acce_z)
                    rawdatalist[0][3].append(gyro_x)
                    rawdatalist[0][4].append(gyro_y)
                    rawdatalist[0][5].append(gyro_z)
                    rawdatalist[0][6].append(ComplementConv(int(listofcontent[index][28:30],16)|(int(listofcontent[index][30:32],16)<<8))/100)
                    rawdatalist[0][7].append(ComplementConv(int(listofcontent[index][32:34],16)|(int(listofcontent[index][34:36],16)<<8))/100)
                elif listofcontent[index][0:4]=="CCCC":
                    Identification="foot"
                    nowf=nowf+datetime.timedelta(seconds=1/samplingfrequency)
                    dt=nowf
                    points ={
                        "measurement": Identification+"initial",
                        "tags": {"id":bingren1},
                        "time": dt,
                        "fields": {
                        "acce_x": acce_x,
                        "acce_y": acce_y,
                        "acce_z": acce_z,
                        "gyro_x": gyro_x,
                        "gyro_y": gyro_y,
                        "gyro_z": gyro_z,
                        "angle_x":ComplementConv(int(listofcontent[index][28:30],16)|(int(listofcontent[index][30:32],16)<<8))/100,
                        "angle_y":ComplementConv(int(listofcontent[index][32:34],16)|(int(listofcontent[index][34:36],16)<<8))/100
                        },
                    }
                    rawdatalist[1][0].append(acce_x)
                    rawdatalist[1][1].append(acce_y)
                    rawdatalist[1][2].append(acce_z)
                    rawdatalist[1][3].append(gyro_x)
                    rawdatalist[1][4].append(gyro_y)
                    rawdatalist[1][5].append(gyro_z)
                    rawdatalist[1][6].append(ComplementConv(int(listofcontent[index][28:30],16)|(int(listofcontent[index][30:32],16)<<8))/100)
                    rawdatalist[1][7].append(ComplementConv(int(listofcontent[index][32:34],16)|(int(listofcontent[index][34:36],16)<<8))/100)
                body.append(points)
                # print(listofcontent[index][4:12])
                # print(index)
            # print(body)
            client.write_points(body)
            hand_datax=np.array(rawdatalist[0][0])
            hand_datay=np.array(rawdatalist[0][1])
            hand_dataz=np.array(rawdatalist[0][2])
            printresults=tremor_detection(np.row_stack((hand_datax,hand_datay,hand_dataz)))
            # print(printresults)
            # print(size(printresults[0]))
            tremor_results=[]
            for m in range(0,len(printresults[0])):
                points ={
                        "measurement": "handtremor",
                        "tags": {"id":bingren1},
                        "time": intial_time_hand+datetime.timedelta(minutes=m),
                        "fields": {
                        "serious_tremor_frequency": printresults[0][m],
                        "mild_tremor_frequency": printresults[1][m],
                        "mild_tremor_level": printresults[2][m],
                        },
                    }
                tremor_results.append(points)
            client.write_points(tremor_results)
            # hand_feature=[[] for _ in range(7)]
            # foot_feature=[[] for _ in range(7)]
            #写入不同的通道
            # for i in range(6):
            hand_feature_acce_x=feature_extraction_func(np.array(rawdatalist[0][0]))
            hand_feature_acce_y=feature_extraction_func(np.array(rawdatalist[0][1]))
            hand_feature_acce_z=feature_extraction_func(np.array(rawdatalist[0][2]))
            hand_feature_gyro_x=feature_extraction_func(np.array(rawdatalist[0][3]))
            hand_feature_gyro_y=feature_extraction_func(np.array(rawdatalist[0][4]))
            hand_feature_gyro_z=feature_extraction_func(np.array(rawdatalist[0][5]))
            hand_feature_angle_x=feature_extraction_func(np.array(rawdatalist[0][6]))
            hand_feature_angle_y=feature_extraction_func(np.array(rawdatalist[0][7]))
            # for i in range(7):
            #      foot_feature[i]=feature_extraction_func(np.array(rawdatalist[1][i]))
            
            for n in range(7): #循环存入7个特征
                hand_time_points=[]
                for times in range(0,hand_feature_acce_x[1][0].shape[0]):
                    points ={
                        "measurement": "hand"+"time"+hand_feature_acce_x[0][n],
                        "tags": {"id":bingren1},
                        "time": intial_time_hand+datetime.timedelta(seconds=times*100/20),
                        "fields": {
                        "acce_x": hand_feature_acce_x[1][n][times],
                        "acce_y": hand_feature_acce_y[1][n][times],
                        "acce_z": hand_feature_acce_z[1][n][times],
                        "gyro_x": hand_feature_gyro_x[1][n][times],
                        "gyro_y": hand_feature_gyro_y[1][n][times],
                        "gyro_z": hand_feature_gyro_z[1][n][times],
                        "angle_x":  hand_feature_angle_x[1][n][times],
                        "angle_y":  hand_feature_angle_y[1][n][times],
                        },
                    }
                    hand_time_points.append(points)
                client.write_points(hand_time_points)
            for n in range(7): #循环存入7个特征
                hand_time_points=[]
                for times in range(0,hand_feature_acce_x[2][0].shape[0]):
                    points ={
                        "measurement": "hand"+"frequency"+hand_feature_acce_x[0][n],
                        "tags": {"id":bingren1},
                        "time": intial_time_hand+datetime.timedelta(seconds=times*100/20),
                        "fields": {
                        "acce_x": hand_feature_acce_x[2][n][times],
                        "acce_y": hand_feature_acce_y[2][n][times],
                        "acce_z": hand_feature_acce_z[2][n][times],
                        "gyro_x": hand_feature_gyro_x[2][n][times],
                        "gyro_y": hand_feature_gyro_y[2][n][times],
                        "gyro_z": hand_feature_gyro_z[2][n][times],
                        "angle_x": hand_feature_angle_x[2][n][times],
                        "angle_y":hand_feature_angle_y[2][n][times],
                        },
                    }
                    hand_time_points.append(points)
                client.write_points(hand_time_points)
        #########################################################         
            foot_feature_acce_x=feature_extraction_func(np.array(rawdatalist[1][0]))
            foot_feature_acce_y=feature_extraction_func(np.array(rawdatalist[1][1]))
            foot_feature_acce_z=feature_extraction_func(np.array(rawdatalist[1][2]))
            foot_feature_gyro_x=feature_extraction_func(np.array(rawdatalist[1][3]))
            foot_feature_gyro_y=feature_extraction_func(np.array(rawdatalist[1][4]))
            foot_feature_gyro_z=feature_extraction_func(np.array(rawdatalist[1][5]))
            foot_feature_angle_x=feature_extraction_func(np.array(rawdatalist[1][6]))
            foot_feature_angle_y=feature_extraction_func(np.array(rawdatalist[1][7]))
            # for i in range(7):
            #      foot_feature[i]=feature_extraction_func(np.array(rawdatalist[1][i]))
            
            for n in range(7): #循环存入7个特征
                foot_time_points=[]
                for times in range(0,foot_feature_acce_x[1][0].shape[0]):
                    points ={
                        "measurement": "foot"+"time"+foot_feature_acce_x[0][n],
                        "tags": {"id":bingren1},
                        "time": intial_time_hand+datetime.timedelta(seconds=times*100/20),
                        "fields": {
                        "acce_x": foot_feature_acce_x[1][n][times],
                        "acce_y": foot_feature_acce_y[1][n][times],
                        "acce_z": foot_feature_acce_z[1][n][times],
                        "gyro_x": foot_feature_gyro_x[1][n][times],
                        "gyro_y": foot_feature_gyro_y[1][n][times],
                        "gyro_z": foot_feature_gyro_z[1][n][times],
                        "angle_x": foot_feature_angle_x[1][n][times],
                        "angle_y": foot_feature_angle_y[1][n][times]
                        },
                    }
                    foot_time_points.append(points)
                client.write_points(foot_time_points)
            for n in range(7): #循环存入7个特征
                foot_time_points=[]
                for times in range(0,foot_feature_acce_x[2][0].shape[0]):
                    points ={
                        "measurement": "foot"+"frequency"+foot_feature_acce_x[0][n],
                        "tags": {"id":bingren1},
                        "time": intial_time_hand+datetime.timedelta(seconds=times*100/20),
                        "fields": {
                        "acce_x": foot_feature_acce_x[2][n][times],
                        "acce_y": foot_feature_acce_y[2][n][times],
                        "acce_z": foot_feature_acce_z[2][n][times],
                        "gyro_x": foot_feature_gyro_x[2][n][times],
                        "gyro_y": foot_feature_gyro_y[2][n][times],
                        "gyro_z": foot_feature_gyro_z[2][n][times],
                        "angle_x": foot_feature_angle_x[2][n][times],
                        "angle_y": foot_feature_angle_y[2][n][times]
                        },
                    }
                    foot_time_points.append(points)
                client.write_points(foot_time_points)
            
        ########################################################
        except Exception as e:
               import traceback
               traceback.print_exc()

        #     raise HTTPException(
        #         status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="查询过程中出现错误，详细信息参见后端日志"
        #     )
        

####################与自动化所测试的，接收TXT文件的接口



@router.post("/uploadfile")
async def create_upload_file(file: UploadFile = File(...)):
    start = time.time()
    contents = await file.read()
    contents1=contents.decode(encoding='gb2312', errors="strict")
    filename=file.filename
    now=filename
    time_rel = datetime.datetime.strptime(now,'%Y/%m/%d %H-%M-%S')
    # td = datetime.timedelta(seconds=time_rel.second)
    # new_dt = time_rel - td
    new_dt = time_rel
    # now=datetime.datetime.fromisoformat(now)
    await queue1.put([contents1,new_dt,filename])
    end = time.time()
    # print("running time:")
    # print(end-start)
    print(len(contents1))
    print(filename)
    # return("ok")
    



##################增加新的程序########################
@router.post("/uploadfile_csv")
async def create_upload_file_csv(file: UploadFile = File(...)):
    # start = time.time()
    contents = await file.read()
    contents1=contents.decode(encoding='utf-8', errors="strict")
    filename=(file.filename)[:-4]
    print(type(filename))
    str_val1 = contents1.splitlines()
    for i in range(len(str_val1)):
        str_val1[i] = np.array(str_val1[i].split(','))
    del str_val1[0]
    samplingfrequency=200
    val1 = np.vstack(str_val1).astype(float)
    now=datetime.datetime.now()
    # strnow = datetime.datetime.strftime(now,'%Y-%m-%d %H:%M:%S') 
    # print(strnow)
    await queue_svm_knn.put([val1,samplingfrequency,now,filename])



########################################################

#worker函数要写到这里面，然后在main里面import进来
queue_svm_knn = asyncio.Queue()
async def workerinput_svm_knn():
    ###部署神经网络算法
    while True:
        receive_data = await queue_svm_knn.get()
        hand_varbel_list, foot_varbel_list, rawoutput=outputfeature_svm_knn(receive_data[0])
        intial_time_hand=receive_data[2]-datetime.timedelta(minutes=int(rawoutput[0][:,0].shape[0]*(1/20)/60))
        client = InfluxDBClient("localhost", 8086, '', '', 'medical')
        bingren1=receive_data[3]
        print(bingren1)
        print(hand_varbel_list)
        print('------------------')
        print(foot_varbel_list)
        print('------------------')
        print(rawoutput[0].shape[0])
        raw_data_hand_feature=[]
        raw_data_hand_amp=[]
        raw_data_hand_freq=[]
        for n in range(0,rawoutput[0].shape[0]):
            acce_x_tremor=0
            acce_y_tremor=0
            acce_z_tremor=0
            acce_x_tremor_amp=0.
            acce_y_tremor_amp=0.
            acce_z_tremor_amp=0.
            acce_x_tremor_freq=0.
            acce_y_tremor_freq=0.
            acce_z_tremor_freq=0.            
            # acce_x_tremor_amp_foot=0.
            # acce_y_tremor_amp_foot=0.
            # acce_z_tremor_amp_foot=0.
            # acce_x_tremor_freq_foot=0.
            # acce_y_tremor_freq_foot=0.
            # acce_z_tremor_freq_foot=0.
            for passageway in hand_varbel_list:
                if int(passageway[0])==0 and (n/20)>=int(passageway[1]) and (n/20)<=int(passageway[2]):
                    acce_x_tremor=1
                    acce_x_tremor_amp=passageway[8]
                    acce_x_tremor_freq=passageway[9]
                if int(passageway[0])==1 and (n/20)>=int(passageway[1]) and (n/20)<=int(passageway[2]):
                    acce_y_tremor=1
                    acce_y_tremor_amp=passageway[8]
                    acce_y_tremor_freq=passageway[9]
                if int(passageway[0])==2 and (n/20)>=int(passageway[1]) and (n/20)<=int(passageway[2]):
                    acce_z_tremor=1
                    acce_z_tremor_amp=passageway[8]
                    acce_z_tremor_freq=passageway[9]
            points ={
                    "measurement": "tremor_time",
                    "tags": {"id":bingren1},
                    "time": intial_time_hand+datetime.timedelta(seconds=n/20),
                    "fields": {
                    "acce_x": acce_x_tremor,
                    "acce_y": acce_y_tremor,
                    "acce_z": acce_z_tremor,
                    },
                    }
            raw_data_hand_feature.append(points)
            points ={
                    "measurement": "tremor_amp",
                    "tags": {"id":bingren1},
                    "time": intial_time_hand+datetime.timedelta(seconds=n/20),
                    "fields": {
                    "acce_x": acce_x_tremor_amp,
                    "acce_y": acce_y_tremor_amp,
                    "acce_z": acce_z_tremor_amp,
                    },
                    }
            raw_data_hand_amp.append(points)
            points ={
                    "measurement": "tremor_freq",
                    "tags": {"id":bingren1},
                    "time": intial_time_hand+datetime.timedelta(seconds=n/20),
                    "fields": {
                    "acce_x": acce_x_tremor_freq,
                    "acce_y": acce_y_tremor_freq,
                    "acce_z": acce_z_tremor_freq,
                    },
                    }
            raw_data_hand_freq.append(points)
        client.write_points(raw_data_hand_feature)
        client.write_points(raw_data_hand_amp)
        client.write_points(raw_data_hand_freq)
        raw_data_foot_feature=[]
        raw_data_foot_amp=[]
        raw_data_foot_freq=[]
        raw_data_foot_speed=[]
        for n in range(0,rawoutput[0].shape[0]):
            acce_x_tremor=0
            acce_y_tremor=0
            acce_z_tremor=0          
            acce_x_tremor_amp=0.
            acce_y_tremor_amp=0.
            acce_z_tremor_amp=0.
            acce_x_tremor_freq=0.
            acce_y_tremor_freq=0.
            acce_z_tremor_freq=0.
            for passageway in foot_varbel_list:
                if int(passageway[0])==0 and (n/20)>=int(passageway[1]) and (n/20)<=int(passageway[2]):
                    acce_x_tremor=1
                    acce_x_tremor_amp=passageway[8]
                    acce_x_tremor_freq=passageway[9]
                if int(passageway[0])==1 and (n/20)>=int(passageway[1]) and (n/20)<=int(passageway[2]):
                    acce_y_tremor=1
                    acce_y_tremor_amp=passageway[8]
                    acce_y_tremor_freq=passageway[9]
                if int(passageway[0])==2 and (n/20)>=int(passageway[1]) and (n/20)<=int(passageway[2]):
                    acce_z_tremor=1
                    acce_z_tremor_amp=passageway[8]
                    acce_z_tremor_freq=passageway[9]
            points ={
                    "measurement": "foot_time",
                    "tags": {"id":bingren1},
                    "time": intial_time_hand+datetime.timedelta(seconds=n/20),
                    "fields": {
                    "acce_x": acce_x_tremor,
                    "acce_y": acce_y_tremor,
                    "acce_z": acce_z_tremor,
                    },
                    }
            raw_data_foot_feature.append(points)
            points ={
                    "measurement": "foot_amp",
                    "tags": {"id":bingren1},
                    "time": intial_time_hand+datetime.timedelta(seconds=n/20),
                    "fields": {
                    "acce_x": acce_x_tremor_amp,
                    "acce_y": acce_y_tremor_amp,
                    "acce_z": acce_z_tremor_amp,
                    },
                    }
            raw_data_foot_amp.append(points)
            points ={
                    "measurement": "foot_freq",
                    "tags": {"id":bingren1},
                    "time": intial_time_hand+datetime.timedelta(seconds=n/20),
                    "fields": {
                    "acce_x": acce_x_tremor_freq,
                    "acce_y": acce_y_tremor_freq,
                    "acce_z": acce_z_tremor_freq,
                    },
                    }
            raw_data_foot_freq.append(points)
            points ={
                    "measurement": "foot_speed",
                    "tags": {"id":bingren1},
                    "time": intial_time_hand+datetime.timedelta(seconds=n/20),
                    "fields": {
                    "acce_x": acce_x_tremor_amp*acce_x_tremor_freq,
                    "acce_y": acce_y_tremor_amp*acce_y_tremor_freq,
                    "acce_z": acce_z_tremor_amp*acce_z_tremor_freq,
                    },
                    }
            raw_data_foot_speed.append(points)
        client.write_points(raw_data_foot_feature)
        client.write_points(raw_data_foot_amp)
        client.write_points(raw_data_foot_freq)
        client.write_points(raw_data_foot_speed)
        hand_feature_acce_x=feature_extraction_func(rawoutput[0][:,0])
        hand_feature_acce_y=feature_extraction_func(rawoutput[0][:,1])
        hand_feature_acce_z=feature_extraction_func(rawoutput[0][:,2])
        hand_feature_gyro_x=feature_extraction_func(rawoutput[1][:,0])
        hand_feature_gyro_y=feature_extraction_func(rawoutput[1][:,1])
        hand_feature_gyro_z=feature_extraction_func(rawoutput[1][:,2])

        foot_feature_acce_x=feature_extraction_func(rawoutput[2][:,0])
        foot_feature_acce_y=feature_extraction_func(rawoutput[2][:,1])
        foot_feature_acce_z=feature_extraction_func(rawoutput[2][:,2])
        foot_feature_gyro_x=feature_extraction_func(rawoutput[3][:,0])
        foot_feature_gyro_y=feature_extraction_func(rawoutput[3][:,1])
        foot_feature_gyro_z=feature_extraction_func(rawoutput[3][:,2])

        raw_data=[]
        for n in range(0,rawoutput[0][:,0].shape[0]):
            acce_x=rawoutput[0][:,0][n]
            acce_y=rawoutput[0][:,1][n]
            acce_z=rawoutput[0][:,2][n]
            gyro_x=rawoutput[1][:,0][n]
            gyro_y=rawoutput[1][:,1][n]
            gyro_z=rawoutput[1][:,2][n]
            points ={
                        "measurement": "handinitial",
                        "tags": {"id":bingren1},
                        "time": intial_time_hand+datetime.timedelta(seconds=n/20),
                        "fields": {
                        "acce_x": acce_x,
                        "acce_y": acce_y,
                        "acce_z": acce_z,
                        "gyro_x": gyro_x,
                        "gyro_y": gyro_y,
                        "gyro_z": gyro_z
                        },
                    }
            raw_data.append(points)
        client.write_points(raw_data)
        for n in range(7): #循环存入7个特征
                hand_time_points=[]
                for times in range(0,hand_feature_acce_x[1][0].shape[0]):
                    points ={
                        "measurement": "hand"+"time"+hand_feature_acce_x[0][n],
                        "tags": {"id":bingren1},
                        "time": intial_time_hand+datetime.timedelta(seconds=times*100/20),
                        "fields": {
                        "acce_x": hand_feature_acce_x[1][n][times],
                        "acce_y": hand_feature_acce_y[1][n][times],
                        "acce_z": hand_feature_acce_z[1][n][times],
                        "gyro_x": hand_feature_gyro_x[1][n][times],
                        "gyro_y": hand_feature_gyro_y[1][n][times],
                        "gyro_z": hand_feature_gyro_z[1][n][times],
                        },
                    }
                    hand_time_points.append(points)
                client.write_points(hand_time_points)
        raw_data_foot=[]
        for n in range(0,rawoutput[0][:,0].shape[0]):
            acce_x=rawoutput[2][:,0][n]
            acce_y=rawoutput[2][:,1][n]
            acce_z=rawoutput[2][:,2][n]
            gyro_x=rawoutput[3][:,0][n]
            gyro_y=rawoutput[3][:,1][n]
            gyro_z=rawoutput[3][:,2][n]
            points ={
                        "measurement": "footinitial",
                        "tags": {"id":bingren1},
                        "time": intial_time_hand+datetime.timedelta(seconds=n/20),
                        "fields": {
                        "acce_x": acce_x,
                        "acce_y": acce_y,
                        "acce_z": acce_z,
                        "gyro_x": gyro_x,
                        "gyro_y": gyro_y,
                        "gyro_z": gyro_z
                        },
                    }
            raw_data_foot.append(points)
        client.write_points(raw_data_foot)
        for n in range(7): #循环存入7个特征
                foot_time_points=[]
                for times in range(0,foot_feature_acce_x[1][0].shape[0]):
                    points ={
                        "measurement": "foot"+"time"+foot_feature_acce_x[0][n],
                        "tags": {"id":bingren1},
                        "time": intial_time_hand+datetime.timedelta(seconds=times*100/20),
                        "fields": {
                        "acce_x": foot_feature_acce_x[1][n][times],
                        "acce_y": foot_feature_acce_y[1][n][times],
                        "acce_z": foot_feature_acce_z[1][n][times],
                        "gyro_x": foot_feature_gyro_x[1][n][times],
                        "gyro_y": foot_feature_gyro_y[1][n][times],
                        "gyro_z": foot_feature_gyro_z[1][n][times],
                        },
                    }
                    foot_time_points.append(points)
                client.write_points(foot_time_points)





######################################################

# # 增加influxdb查询某一病人最近一段时间的分类数据的功能，并以列表返回
class user(BaseModel):
    username: str
    selectdataset: str
    characteristicset:str
    pasttime: list
    grouptime: str

@router.post("/find_result_data")
async def find_result_patients(item: user, current_user: JWTUser = Depends(get_current_user),
influxclient: InfluxDBClient = Depends(get_influxdb_connection),):
    print(item.username)
    print(item.selectdataset)
    print(item.characteristicset)
    print(item.pasttime[0])
    print(item.pasttime[1])
    print(item.grouptime)
    item.pasttime[0]=datetime.datetime.strftime(datetime.datetime.strptime(item.pasttime[0], "%Y-%m-%dT%H:%M:%S.%fZ")+datetime.timedelta(hours=8) ,'%Y-%m-%d %H:%M:%S')
    item.pasttime[1]=datetime.datetime.strftime(datetime.datetime.strptime(item.pasttime[1], "%Y-%m-%dT%H:%M:%S.%fZ")+datetime.timedelta(hours=8) ,'%Y-%m-%d %H:%M:%S')
    # if item.characteristicset=="initial":
    #     fields=item.selectdataset 
    # else:
    #     fields=item.characteristicset
    try:
        # client = InfluxDBClient('localhost', 8086, 'lihui', '123456')
        # if item.characteristicset=="initial":
        #     sql=f"SELECT {item.selectdataset} FROM medical.autogen.hand{item.characteristicset} WHERE time > '{item.pasttime[0]}' AND time < '{item.pasttime[1]}' AND id='{item.username}'"
        # else:
        sql=f"SELECT {item.selectdataset} FROM medical.autogen.{item.characteristicset} WHERE time > '{item.pasttime[0]}' AND time < '{item.pasttime[1]}' AND id='{item.username}'"
        # sql = f"SELECT mean({fields}) AS output FROM medical.autogen.{tags} WHERE time > '{item.pasttime[0]}' AND time < '{item.pasttime[1]}' AND id='{item.username}' GROUP BY time({item.grouptime}) FILL(none)"
        print(sql)
        result = influxclient.query(sql)
        points = result.get_points()
        outputinfo = []
        time = []
        # print(result)
        if result.items():
            # num25=0
            # num50=0
            # num75=0
            # num100=0
            for eitem in points:  # 这条用于循环取出某一项，比如时间“time”或者加速度“acce_x0”
                # print(eitem["acce_x"])
                outputinfo.append(eitem[item.selectdataset])
                # if item["output"]<=25:
                #     num25=num25+1
                # elif 25<item["output"]<=50:
                #     num50=num50+1
                # elif 50<item["output"]<=75:
                #     num75=num75+1
                # else:
                #     num100=num100+1
                # utc_date = datetime.datetime.strptime(eitem['time'], "%Y-%m-%dT%H:%M:%SZ")
                # local_date = utc_date + datetime.timedelta(hours=8)
                # local_date_str = datetime.datetime.strftime(local_date ,'%Y-%m-%d %H:%M:%S')
                time.append(eitem['time'])
            return({'output': outputinfo, "time": time})
        else:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="该用户不存在或者该时间段没有数据")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="查询过程中出现错误，详细信息参见后端日志"
        )



#####################################################




####################与自动化所测试的，接收TXT文件的接口


#####转换函数######
def ComplementConv(DT):
    Raw = 0
    if DT & 0x8000 == 0x8000:
        Raw = -((~ DT & 0x7FFF) + 1)
    else:
        Raw = DT
    return Raw
#####转换函数######




# 增加queue排队算法，以及神经网络接口


# class transmitdata(BaseModel):
#     username: str
#     senddata: list
#     timestamp: str


# @router.post("/transmit_data")
# async def transmit_data(item: transmitdata):
#     tz_utc_8 = datetime.timezone(datetime.timedelta(hours=8))
#     dt=datetime.datetime.strptime(item.timestamp,"%Y-%m-%d %H:%M:%S")
#     dt = dt.replace(tzinfo=tz_utc_8)
#     # print(item.senddata)
#     await queue.put([item.senddata,dt,item.username])


# 增加与中科院自动化所测试的接口

class transmitdatafeeling(BaseModel):
    姓名: str
    年龄: str
    性别: str
    电话: str
    身份证号: str
    家庭住址: str
    首发症状时间: str
    运动症状: str
    非运动症状: str
    确诊医院: str
    目前负责医生: str
    开关现象: str
    DBS手术治疗: str
    美多巴: str
    柯丹: str
    森福罗: str
    罗替高汀贴剂: str
    金刚烷胺:str

@router.post("/transmit_feeling")
async def transmit_feeling(item: transmitdatafeeling,mongodb_client: MongoClient = Depends(get_mongodb_connection)):
    myquery = { "idcard":item.身份证号}
    newvalues = { "$set": {"first_symptom_time": item.首发症状时间,"motor_symptoms":item.运动症状,"nonmotor_symptoms":item.非运动症状,"hosptial":item.确诊医院,
    "responsibility_doctor":item.目前负责医生,"onoff_phenomenon":item.开关现象,"DBS":item.DBS手术治疗,"medication":[item.美多巴,item.森福罗,item.柯丹,item.罗替高汀贴剂,
    item.金刚烷胺]}}
    mydb = mongodb_client["ccs"]
    mycol = mydb["users"]
    mycol.update_one(myquery, newvalues)
    # print(item.姓名)
    # print(item.年龄)
    # print(item.性别)
    # print(item.电话)
    # print(item.身份证号)
    # print(item.家庭住址)
    # print(item.首发症状时间)
    # print(item.运动症状)
    # print(item.非运动症状)
    # print(item.确诊医院)
    # print(item.目前负责医生)
    # print(item.开关现象)
    # print(item.DBS手术治疗)
    # print(item.美多巴)
    # print(item.柯丹)
    # print(item.森福罗)
    # print(item.罗替高汀贴剂)
    # print(item.金刚烷胺)
    

class testtransmitdata(BaseModel):
    name: str
    distinguish:str
    time: str
    acce_x:str
    acce_y:str
    acce_z:str
    gyro_x:str
    gyro_y:str
    gyro_z:str
    # timestamp: str


@router.post("/transmit_data_test")
async def transmit_data_test(item: testtransmitdata,influxclient: InfluxDBClient = Depends(get_influxdb_connection),):
    # print(item.username)
    # print(item.senddata)
    # aaaaa=int(item.acce_x,base=16),
    body = [{
         "measurement": item.distinguish,
         "tags": {"id": item.name},
        #  "time":item.time,
         "fields": {
            "acce_x": round(int(item.acce_y,16)*(8*9.86/32768),4),
            "acce_y": round(int(item.acce_y,16)*(8*9.86/32768),4),
            "acce_z": round(int(item.acce_z,16)*(8*9.86/32768),4),
            "gyro_x": round(int(item.gyro_x,16)*(2000/(32768*(3.14/180))),4),
            "gyro_y": round(int(item.gyro_y,16)*(2000/(32768*(3.14/180))),4),
            "gyro_z": round(int(item.gyro_z,16)*(2000/(32768*(3.14/180))),4),
            "timestamp": int(item.time,16),
            },
           }
          ]
    influxclient.write_points(body)
    body1= [{
         "measurement": item.distinguish+"realdata",
         "tags": {"id": item.name},
        #  "time":item.time,
         "fields": {
            "acce_x": item.acce_y,
            "acce_y": item.acce_y,
            "acce_z": item.acce_z,
            "gyro_x": item.gyro_x,
            "gyro_y": item.gyro_y,
            "gyro_z": item.gyro_z,
            "timestamp": int(item.time,16),
            },
           }
          ]
    influxclient.write_points(body1)









#这个地方必须要直接返回Response(content=msg,media_type="application/text")，否则，返回数据带双引号，平台认证会失败。
@router.get("/transmit_data_test_iot")
async def transmit_data_test_iot_get(msg,nonce:Optional[str] = None,signature:Optional[str] = None):
      print(type(msg))
    #   print(type(eval(msg)))
      return Response(content=msg,media_type="application/text")
#这个地方必须要直接返回Response(content=msg,media_type="application/text")，否则，返回数据带双引号，平台认证会失败。



class transmitdataiot(BaseModel):
    msg:dict
    msg_signature:str
    nonce:str


@router.post("/transmit_data_test_iot")
async def transmit_data_test_iot_post(item: transmitdataiot,influxclient: InfluxDBClient = Depends(get_influxdb_connection),):
    print(item.msg)
    print(item.msg_signature)
    print(item.nonce)
    # aaaaa=int(item.acce_x,base=16),
    # body = [{
    #      "measurement": item.distinguish,
    #      "tags": {"id": item.name},
    #     #  "time":item.time,
    #      "fields": {
    #         "acce_x": round(int(item.acce_y,16)*(8*9.86/32768),4),
    #         "acce_y": round(int(item.acce_y,16)*(8*9.86/32768),4),
    #         "acce_z": round(int(item.acce_z,16)*(8*9.86/32768),4),
    #         "gyro_x": round(int(item.gyro_x,16)*(2000/(32768*(3.14/180))),4),
    #         "gyro_y": round(int(item.gyro_y,16)*(2000/(32768*(3.14/180))),4),
    #         "gyro_z": round(int(item.gyro_z,16)*(2000/(32768*(3.14/180))),4),
    #         "timestamp": int(item.time,16),
    #         },
    #        }
    #       ]




    # tz_utc_8 = datetime.timezone(datetime.timedelta(hours=8))
    # dt=datetime.datetime.strptime(item.timestamp,"%Y-%m-%d %H:%M:%S")
    # dt = dt.replace(tzinfo=tz_utc_8)
    # print(item.senddata)
    # await queue.put([item.senddata,dt,item.username])

   # await queue.put([item.senddata,dt,item.username])




# 根据用户名查询单个病人数据
class finduser(BaseModel):
    UserInfo: str


@router.post("/finduserinfo")
async def finduserinfo(
    searchuser_info: finduser,
    mongodb_client: MongoClient = Depends(get_mongodb_connection),
):
    """
    搜索用户
    """
    db = mongodb_client["ccs"]
    user_collection = db["users"]
    user_dict = []
    try:
        db_user = user_collection.find(
            {"name": searchuser_info.UserInfo, "role": "user"})
        if db_user is None:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST, detail="没有这样的用户！")
        else:
            result_list = list(db_user[:])
            for document in result_list:
                mydict = {}
                mydict["name"] = format(document["name"])
                mydict["username"] = format(document["username"])
                mydict["age"] = format(document["age"])
                mydict["gender"] = format(document["gender"])
                mydict["phone"] = format(document["phone"])
                mydict["idcard"] = format(document["idcard"])
                mydict["email"] = format(document["email"])
                mydict["address"] = format(document["address"])
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


# 根据授权用户名直接显示单个用户的数据(貌似没有用)
@router.get("/authorize_represent_data/")
async def authorize_represent_data(
    current_user: JWTUser = Depends(get_current_user),
    mongodb_client: MongoClient = Depends(get_mongodb_connection),
    ):
    # mclient = MongoClient(host=MONGODB_HOST, port=27017)
    mydb = mongodb_client["ccs"]
    mycol = mydb["users"]
    mydoc = mycol.find_one({"id": current_user.id})
    del mydoc["_id"]
    return mydoc




# 在医生端口通过主治医师来查询当前所对应的所有病人（5月14日改进版本）


@router.get("/find_all_patients/")
async def get_all_patients(current_user: JWTUser = Depends(get_current_user),
mongodb_client: MongoClient = Depends(get_mongodb_connection),):
    # mclient = MongoClient(host=MONGODB_HOST, port=27017)
    mydb = mongodb_client["ccs"]
    mycol = mydb["users"]
    try:
        mydoc1 = mycol.find_one(
            {"username": current_user.username}, {"patientslist": 1})
        if mydoc1['patientslist']:
            userinformation = []
            for userinform in mycol.find({"username": {"$in": mydoc1['patientslist']}}):
                userinformation.append({'name': userinform["name"], 'age': userinform["age"], 'gender': userinform["gender"],
                                        'address': userinform["address"], 'username': userinform["username"]})
            return(userinformation)
        else:
            return("nopatients")
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="查询过程中出现错误，详细信息参见后端日志"
        )


# 回复病人主观问卷的api接口（11月11日改进版本）


@router.get("/subjective_questionnaire")
async def subjective_questionnaire(current_user: JWTUser = Depends(get_current_user),
mongodb_client: MongoClient = Depends(get_mongodb_connection),):
    # mclient = MongoClient(host=MONGODB_HOST, port=27017)
    mydb = mongodb_client["ccs"]
    mycol = mydb["users"]
    try:
        mydoc1 = mycol.find_one(
            {"username": current_user.username}, {"patientslist": 1})
        if mydoc1['patientslist']:
            userinformation = []
            for userinform in mycol.find({"username": {"$in": mydoc1['patientslist']}}):
                userinformation.append({'name': userinform["name"], 'age': userinform["age"], 'gender': userinform["gender"],
                                        'address': userinform["address"], 'username': userinform["username"],
                                        "questionnaire":{"first_symptom_time":userinform["first_symptom_time"],
                                        'motor_symptoms':userinform['motor_symptoms'],'nonmotor_symptoms':userinform['nonmotor_symptoms'],'hosptial':userinform["hosptial"],
                                        'responsibility_doctor':userinform['responsibility_doctor'],"onoff_phenomenon":userinform['onoff_phenomenon'],"DBS":userinform['DBS'],
                                        "medication":{'Medopar':userinform['medication'][0],"Comtan":userinform['medication'][2],
                                        'Sifrol':userinform['medication'][1],"Rotigotine":userinform['medication'][3],'Amantadine':userinform['medication'][4]}}
                                        })
            return(userinformation)
        else:
            return("nopatients")
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="查询过程中出现错误，详细信息参见后端日志"
        )


# 根据token向前端发送患者的个人信息（5月14日改进版本）
@router.get("/finduserinformation/{role}")
async def get_all_patients(role: str, current_user: JWTUser = Depends(get_current_user),
mongodb_client: MongoClient = Depends(get_mongodb_connection),):
    # mclient = MongoClient(host=MONGODB_HOST, port=27017)
    mydb = mongodb_client["ccs"]
    mycol = mydb["users"]
    try:
        if role == "patience":
            mydoc1 = mycol.find_one({"username": current_user.username}, {
                                    "name": 1, "age": 1, "gender": 1, "address": 1, "email": 1, "phone": 1, "idcard": 1})
            del mydoc1["_id"]
            return mydoc1
        elif role == "doctor":
            mydoc1 = mycol.find_one({"username": current_user.username}, {"name": 1, "age": 1, "gender": 1,
                                                                          "address": 1, "email": 1, "phone": 1, "idcard": 1, "company": 1, "professionalranks": 1})
            del mydoc1["_id"]
            return mydoc1
        #  return mydoc1
        #  return(mydoc1)
    #   if mydoc1['userinformation']:
    #      return(mydoc1['userinformation'])
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="查询过程中出现错误，详细信息参见后端日志"
        )


# 插入数据/修改数据，主要帮助病人填写主治医师5-18新编写的


class Insertuserinfo(BaseModel):
    name: str
    age: str
    gender: str
    address: str
    email: str
    phone: str
    idcard: str
    company: str = None
    professionalranks: str = None


@router.post("/change_userinfo")
async def change_userinfo(item: Insertuserinfo, current_user: JWTUser = Depends(get_current_user),
mongodb_client: MongoClient = Depends(get_mongodb_connection),):
    try:
        item_dict = item.dict()
        # mclient = MongoClient(host=MONGODB_HOST, port=27017)
        mydb = mongodb_client["ccs"]
        mycol = mydb["users"]
        mycol.update_one({"username": current_user.username}, {"$set": item_dict})
        return("success")
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="查询过程中出现错误，详细信息参见后端日志"
        )

# 二次注册补全病人的meta信息，5-22新编写的5-27再次修改
class twiceinsertinfo(BaseModel):
    name: str
    age: str
    gender: str
    address: str


@router.post("/medical_patiencesignup")
async def medical_signup2(twifo: twiceinsertinfo, current_user: JWTUser = Depends(get_current_user),
mongodb_client: MongoClient = Depends(get_mongodb_connection),):
    try:
        twifo_dict = twifo.dict()
        # mclient = MongoClient(host=MONGODB_HOST, port=27017)
        mydb = mongodb_client["ccs"]
        mycol = mydb["users"]
        twifo_dict.update({"medicalsignup": "Y", "role": "user","prescriptionlist":[],"first_symptom_time":"-","motor_symptoms":"-"
        ,"nonmotor_symptoms":"-","hosptial":"-","responsibility_doctor":"-","onoff_phenomenon":"-","DBS":"-"
        ,'medication':['-','-','-','-','-']})
        mycol.update_one({"username": current_user.username}, {"$set": twifo_dict})
        return("success")
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="查询过程中出现错误，详细信息参见后端日志"
        )


# 二次注册补全病人的meta信息，5-27编写的
class twiceinsertdoctorinfo(BaseModel):
    name: str
    age: str
    gender: str
    address: str
    company: str
    professionalranks: str
    vericode: str


@router.post("/medical_doctorsignup")
async def medical_doctorsignup2(twifo: twiceinsertdoctorinfo, current_user: JWTUser = Depends(get_current_user),
mongodb_client: MongoClient = Depends(get_mongodb_connection),):
    try:
        twifo_dict = twifo.dict()
        if twifo_dict['vericode'] == "doctorsignup":
            del twifo_dict['vericode']
            # mclient = MongoClient(host=MONGODB_HOST, port=27017)
            mydb = mongodb_client["ccs"]
            mycol = mydb["users"]
            twifo_dict.update({"medicalsignup": "Y", "role": "admin", "patientslist": []})
            mycol.update_one({"username": current_user.username}, {"$set": twifo_dict})
            return("success")
        else:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST, detail="验证码错误")
    except HTTPException:
        raise
    except Exception as e:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="查询过程中出现错误，详细信息参见后端日志"
        )


# 修改密码5-19新编写的
class change_password(BaseModel):
    pwdInput: str
    pwdNew: str


@router.post("/change_pwd")
async def changepassword(pwd: change_password, current_user: JWTUser = Depends(get_current_user),
mongodb_client: MongoClient = Depends(get_mongodb_connection),):
    try:
        pwd_dict = pwd.dict()
        # mclient = MongoClient(host=MONGODB_HOST, port=27017)
        mydb = mongodb_client["ccs"]
        mycol = mydb["users"]
        check_password = mycol.find_one(
            {"username": current_user.username, "password": pwd.pwdInput})
        if check_password is None:
            return "pwdnotright"
        else:
            mycol.update_one({"username": current_user.username}, {
                             "$set": {"password": pwd.pwdNew}})
            return("success")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="查询过程中出现错误，详细信息参见后端日志"
        )

# 医生端口增加监管的病人5-31新编
class addusername(BaseModel):
    username: str


@router.post("/addpatients")
async def addpatients(adduser: addusername, current_user: JWTUser = Depends(get_current_user),
mongodb_client: MongoClient = Depends(get_mongodb_connection),):
    try:
        # mclient = MongoClient(host=MONGODB_HOST, port=27017)
        mydb = mongodb_client["ccs"]
        mycol = mydb["users"]
        patience = mycol.find_one({"username": adduser.username})
        if patience:
            mycol.update_one({"username": current_user.username}, {
                             "$push": {"patientslist": adduser.username}})
            return("success")
        else:
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST, detail="用户未找到！")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="查询过程中出现错误，详细信息参见后端日志"
        )


# 医生端口删除监管的病人5-31新编
class deleteusername(BaseModel):
    username: str


@router.post("/deletepatients")
async def deletepatients(deleteuser: deleteusername, current_user: JWTUser = Depends(get_current_user),
mongodb_client: MongoClient = Depends(get_mongodb_connection),):
    try:
        # mclient = MongoClient(host=MONGODB_HOST, port=27017)
        mydb = mongodb_client["ccs"]
        mycol = mydb["users"]
        mycol.update_one({"username": current_user.username}, {
                         "$pull": {"patientslist": deleteuser.username}})
        return("success")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="查询过程中出现错误，详细信息参见后端日志"
        )






# # # 增加influxdb查询某一病人最近一段时间的加速度数据的功能，并以列表返回，这个先注释掉，需要再打开
# class user(BaseModel):
#     username: str
#     selectdataset: str
#     pasttime: str
#     grouptime: str

# @router.post("/find_acceleration_data")
# async def find_acceleration_patients(item: user, current_user: JWTUser = Depends(get_current_user)):
#     # print(item.selectdataset)
#     try:
#         client = InfluxDBClient('localhost', 8086, 'lihui', '123456')
#         sql = f"SELECT mean({item.selectdataset}) AS output FROM medical.autogen.bingrenshuju WHERE time > now() - {item.pasttime} AND time < now() AND id='{item.username}' GROUP BY time({item.grouptime})"
#         # print(sql)
#         result = client.query(sql)
#         points = result.get_points()
#         outputinfo = []
#         time = []
#         if result.items():
#             for item in points:  # 这条用于循环取出某一项，比如时间“time”或者加速度“acce_x0”
#                 outputinfo.append(item['output'])
#                 utc_date = datetime.datetime.strptime(item['time'], "%Y-%m-%dT%H:%M:%SZ")
#                 local_date = utc_date + datetime.timedelta(hours=8)
#                 local_date_str = datetime.datetime.strftime(local_date ,'%Y-%m-%d %H:%M:%S')
#                 # print(local_date_str)
#                 time.append(local_date_str)
#                 # print(type(item['output']))
#             return({'output': outputinfo, "time": time})
#         else:
#             raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="该用户不存在或者该时间段没有数据")
#     except HTTPException:
#         raise
#     except Exception as e:
#         import traceback
#         traceback.print_exc()
#         raise HTTPException(
#             status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="查询过程中出现错误，详细信息参见后端日志"
#         )


# # 增加influxdb查询某一病人最近一段时间的分类数据的功能，并以列表返回
class user(BaseModel):
    username: str
    selectdataset: str
    characteristicset:str
    pasttime: list
    grouptime: str

@router.post("/find_acceleration_data")
async def find_acceleration_patients(item: user, current_user: JWTUser = Depends(get_current_user),
influxclient: InfluxDBClient = Depends(get_influxdb_connection),):
    print(item.username)
    print(item.selectdataset)
    print(item.characteristicset)
    print(item.pasttime[0])
    print(item.pasttime[1])
    print(item.grouptime)
    item.pasttime[0]=datetime.datetime.strftime(datetime.datetime.strptime(item.pasttime[0], "%Y-%m-%dT%H:%M:%S.%fZ")+datetime.timedelta(hours=8) ,'%Y-%m-%d %H:%M:%S')
    item.pasttime[1]=datetime.datetime.strftime(datetime.datetime.strptime(item.pasttime[1], "%Y-%m-%dT%H:%M:%S.%fZ")+datetime.timedelta(hours=8) ,'%Y-%m-%d %H:%M:%S')
    # if item.characteristicset=="initial":
    #     fields=item.selectdataset 
    # else:
    #     fields=item.characteristicset
    try:
        # client = InfluxDBClient('localhost', 8086, 'lihui', '123456')
        if item.characteristicset=="initial":
            sql=f"SELECT {item.selectdataset} FROM medical.autogen.hand{item.characteristicset} WHERE time > '{item.pasttime[0]}' AND time < '{item.pasttime[1]}' AND id='{item.username}'"
        else:
            sql=f"SELECT {item.selectdataset} FROM medical.autogen.handtime{item.characteristicset} WHERE time > '{item.pasttime[0]}' AND time < '{item.pasttime[1]}' AND id='{item.username}'"
        # sql = f"SELECT mean({fields}) AS output FROM medical.autogen.{tags} WHERE time > '{item.pasttime[0]}' AND time < '{item.pasttime[1]}' AND id='{item.username}' GROUP BY time({item.grouptime}) FILL(none)"
        print(sql)
        result = influxclient.query(sql)
        points = result.get_points()
        outputinfo = []
        time = []
        # print(result)
        if result.items():
            # num25=0
            # num50=0
            # num75=0
            # num100=0
            for eitem in points:  # 这条用于循环取出某一项，比如时间“time”或者加速度“acce_x0”
                # print(eitem["acce_x"])
                outputinfo.append(eitem[item.selectdataset])
                # if item["output"]<=25:
                #     num25=num25+1
                # elif 25<item["output"]<=50:
                #     num50=num50+1
                # elif 50<item["output"]<=75:
                #     num75=num75+1
                # else:
                #     num100=num100+1
                # utc_date = datetime.datetime.strptime(eitem['time'], "%Y-%m-%dT%H:%M:%SZ")
                # local_date = utc_date + datetime.timedelta(hours=8)
                # local_date_str = datetime.datetime.strftime(local_date ,'%Y-%m-%d %H:%M:%S')
                time.append(eitem['time'])
            return({'output': outputinfo, "time": time})
        else:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="该用户不存在或者该时间段没有数据")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="查询过程中出现错误，详细信息参见后端日志"
        )


# # 增加influxdb查询某一病人最近一段时间的分类数据的功能，并以列表返回
class user(BaseModel):
    username: str
    selectdataset: str
    characteristicset:str
    pasttime: list
    grouptime: str

@router.post("/find_foot_acceleration_data")
async def find_foot_acceleration_patients(item: user, current_user: JWTUser = Depends(get_current_user),
influxclient: InfluxDBClient = Depends(get_influxdb_connection),):
    print(item.username)
    print(item.selectdataset)
    print(item.characteristicset)
    print(item.pasttime[0])
    print(item.pasttime[1])
    print(item.grouptime)
    item.pasttime[0]=datetime.datetime.strftime(datetime.datetime.strptime(item.pasttime[0], "%Y-%m-%dT%H:%M:%S.%fZ")+datetime.timedelta(hours=8) ,'%Y-%m-%d %H:%M:%S')
    item.pasttime[1]=datetime.datetime.strftime(datetime.datetime.strptime(item.pasttime[1], "%Y-%m-%dT%H:%M:%S.%fZ")+datetime.timedelta(hours=8) ,'%Y-%m-%d %H:%M:%S')
    # if item.characteristicset=="initial":
    #     fields=item.selectdataset 
    # else:
    #     fields=item.characteristicset
    try:
        # client = InfluxDBClient('localhost', 8086, 'lihui', '123456')
        if item.characteristicset=="initial":
            sql=f"SELECT {item.selectdataset} FROM medical.autogen.foot{item.characteristicset} WHERE time > '{item.pasttime[0]}' AND time < '{item.pasttime[1]}' AND id='{item.username}'"
        else:
            sql=f"SELECT {item.selectdataset} FROM medical.autogen.foottime{item.characteristicset} WHERE time > '{item.pasttime[0]}' AND time < '{item.pasttime[1]}' AND id='{item.username}'"
        # sql = f"SELECT mean({fields}) AS output FROM medical.autogen.{tags} WHERE time > '{item.pasttime[0]}' AND time < '{item.pasttime[1]}' AND id='{item.username}' GROUP BY time({item.grouptime}) FILL(none)"
        print(sql)
        result = influxclient.query(sql)
        points = result.get_points()
        outputinfo = []
        time = []
        # print(result)
        if result.items():
            # num25=0
            # num50=0
            # num75=0
            # num100=0
            for eitem in points:  # 这条用于循环取出某一项，比如时间“time”或者加速度“acce_x0”
                # print(eitem["acce_x"])
                outputinfo.append(eitem[item.selectdataset])
                # if item["output"]<=25:
                #     num25=num25+1
                # elif 25<item["output"]<=50:
                #     num50=num50+1
                # elif 50<item["output"]<=75:
                #     num75=num75+1
                # else:
                #     num100=num100+1
                # utc_date = datetime.datetime.strptime(eitem['time'], "%Y-%m-%dT%H:%M:%SZ")
                # local_date = utc_date + datetime.timedelta(hours=8)
                # local_date_str = datetime.datetime.strftime(local_date ,'%Y-%m-%d %H:%M:%S')
                time.append(eitem['time'])
            return({'output': outputinfo, "time": time})
        else:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="该用户不存在或者该时间段没有数据")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="查询过程中出现错误，详细信息参见后端日志"
        )


# # 增加influxdb查询某一病人最近一段时间的分类数据的功能，并以列表返回
class user(BaseModel):
    username: str
    selectdataset: str
    characteristicset:str
    pasttime: list
    grouptime: str

@router.post("/find_tremor_acceleration_data")
async def find_tremor_acceleration_patients(item: user, current_user: JWTUser = Depends(get_current_user),
influxclient: InfluxDBClient = Depends(get_influxdb_connection),):
    print(item.username)
    print(item.selectdataset)
    print(item.characteristicset)
    print(item.pasttime[0])
    print(item.pasttime[1])
    print(item.grouptime)
    item.pasttime[0]=datetime.datetime.strftime(datetime.datetime.strptime(item.pasttime[0], "%Y-%m-%dT%H:%M:%S.%fZ")+datetime.timedelta(hours=8) ,'%Y-%m-%d %H:%M:%S')
    item.pasttime[1]=datetime.datetime.strftime(datetime.datetime.strptime(item.pasttime[1], "%Y-%m-%dT%H:%M:%S.%fZ")+datetime.timedelta(hours=8) ,'%Y-%m-%d %H:%M:%S')
    # if item.characteristicset=="initial":
    #     fields=item.selectdataset 
    # else:
    #     fields=item.characteristicset
    try:
        # client = InfluxDBClient('localhost', 8086, 'lihui', '123456')
        sql=f"SELECT {item.selectdataset} FROM medical.autogen.handtremor WHERE time > '{item.pasttime[0]}' AND time < '{item.pasttime[1]}' AND id='{item.username}'"
        print(sql)
        result = influxclient.query(sql)
        points = result.get_points()
        outputinfo = []
        time = []
        # print(result)
        if result.items():
            # num25=0
            # num50=0
            # num75=0
            # num100=0
            for eitem in points:  # 这条用于循环取出某一项，比如时间“time”或者加速度“acce_x0”
                # print(eitem["acce_x"])
                outputinfo.append(eitem[item.selectdataset])
                # if item["output"]<=25:
                #     num25=num25+1
                # elif 25<item["output"]<=50:
                #     num50=num50+1
                # elif 50<item["output"]<=75:
                #     num75=num75+1
                # else:
                #     num100=num100+1
                # utc_date = datetime.datetime.strptime(eitem['time'], "%Y-%m-%dT%H:%M:%SZ")
                # local_date = utc_date + datetime.timedelta(hours=8)
                # local_date_str = datetime.datetime.strftime(local_date ,'%Y-%m-%d %H:%M:%S')
                time.append(eitem['time'])
            return({'output': outputinfo, "time": time})
        else:
            raise HTTPException(status_code=HTTP_400_BAD_REQUEST, detail="该用户不存在或者该时间段没有数据")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="查询过程中出现错误，详细信息参见后端日志"
        )


# 增加向病人账户增添处方的功能
class addprescription(BaseModel):
    username: str
    prescriptionFrom: list

@router.post("/addprescription")
async def addprescription(prescription: addprescription, current_user: JWTUser = Depends(get_current_user),
mongodb_client: MongoClient = Depends(get_mongodb_connection),):
    try:
        # mclient = MongoClient(host=MONGODB_HOST, port=27017)
        mydb = mongodb_client["ccs"]
        mycol = mydb["users"]
        mycol.update_one({"username": prescription.username}, {
                         "$push": {"prescriptionlist": prescription.prescriptionFrom}})
        return("success")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="查询过程中出现错误，详细信息参见后端日志"
        )

# 增加拉取某一用户所有处方的的功能
class requestprescription(BaseModel):
    username: str

@router.post("/requestprescription")
async def requestprescription(request: requestprescription, current_user: JWTUser = Depends(get_current_user),
mongodb_client: MongoClient = Depends(get_mongodb_connection),):
    try:
        # mclient = MongoClient(host=MONGODB_HOST, port=27017)
        mydb = mongodb_client["ccs"]
        mycol = mydb["users"]
        prescriptionlists=mycol.find_one({"username": request.username}, {"prescriptionlist": 1})['prescriptionlist']
        user_dict=[]
        for document in prescriptionlists:
            mydict = {}
            mydict["name"] = format(document[0])
            mydict["date"] = format(document[1])
            mydict["text"] = format(document[2])
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



            
# 医生端口删除监管的病人5-31新编
class deletePrescription(BaseModel):
    username:str
    prescription: list


@router.post("/deletePrescription")
async def deletePrescription(deletePrescription: deletePrescription, current_user: JWTUser = Depends(get_current_user),
mongodb_client: MongoClient = Depends(get_mongodb_connection),):
    try:
        # mclient = MongoClient(host=MONGODB_HOST, port=27017)
        mydb = mongodb_client["ccs"]
        mycol = mydb["users"]
        mycol.update_one({"username": deletePrescription.username}, {
                         "$pull": {"prescriptionlist":deletePrescription.prescription}})
        print(deletePrescription.prescription)
        return("success")
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR, detail="查询过程中出现错误，详细信息参见后端日志"
        )