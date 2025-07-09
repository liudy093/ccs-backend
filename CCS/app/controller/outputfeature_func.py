# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 11:06:10 2020

@author: Administrator
"""
# import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
# import seaborn as sns
# import os
import joblib
from app.controller.feature_extraction_func import feature_extraction_func

def outputfeature(data:str):
    # f = open(inputset,"r")
    #####在这里输入文件名
    # print(type(data[0]))
    # print(data[0])
    # data= f.read()
    # print(data)
    # print(len(data[0]))
    # print(data[0][1:2])
    
    listofcontent=[]
    
    i=0
    while i<=(len(data)-40):
        if (data[i:i+4]=="AAAA") and (data[i+36:i+40]=="FFBB"):
           listofcontent.append(data[i:i+40])
           i=i+40
        elif (data[i:i+4]=="CCCC") and (data[i+36:i+40]=="FFBB"):
           listofcontent.append(data[i:i+40])
           i=i+40
        else:
           i=i+1 
    # print(listofcontent)
    
    
    
    #####转换函数######
    def ComplementConv(DT):
        Raw = 0
        if DT & 0x8000 == 0x8000:
            Raw = -((~ DT & 0x7FFF) + 1)
        else:
            Raw = DT
        return Raw
    #####转换函数######
    # hand_timestamp=[]
    hand_acce_x=[]
    hand_acce_y=[]
    hand_acce_z=[]
    hand_gyro_x=[]
    hand_gyro_y=[]
    hand_gyro_z=[]
    hand_angle_listx=[]
    hand_angle_listy=[]
    ##################
    # foot_timestamp=[]
    foot_acce_x=[]
    foot_acce_y=[]
    foot_acce_z=[]
    foot_gyro_x=[]
    foot_gyro_y=[]
    foot_gyro_z=[]
    # foot_angle=[]
    foot_angle_listx=[]
    foot_angle_listy=[]
    # sumlist=[]
    for index in range(0,len(listofcontent)):
            acce_x=ComplementConv(int(listofcontent[index][4:6],16)|(int(listofcontent[index][6:8],16)<<8))*(8*9.86/32768)
            acce_y=ComplementConv(int(listofcontent[index][8:10],16)|(int(listofcontent[index][10:12],16)<<8))*(8*9.86/32768)
            acce_z=ComplementConv(int(listofcontent[index][12:14],16)|(int(listofcontent[index][14:16],16)<<8))*(8*9.86/32768)
            gyro_x=ComplementConv(int(listofcontent[index][16:18],16)|(int(listofcontent[index][18:20],16)<<8))*(2000/32768*(3.14/180))
            gyro_y=ComplementConv(int(listofcontent[index][20:22],16)|(int(listofcontent[index][22:24],16)<<8))*(2000/32768*(3.14/180))
            gyro_z=ComplementConv(int(listofcontent[index][24:26],16)|(int(listofcontent[index][26:28],16)<<8))*(2000/32768*(3.14/180))
            if listofcontent[index][0:4]=="AAAA":
                  hand_acce_x.append(acce_x)
                  hand_acce_y.append(acce_y)
                  hand_acce_z.append(acce_z)
                  hand_gyro_x.append(gyro_x)
                  hand_gyro_y.append(gyro_y)
                  hand_gyro_z.append(gyro_z)
                  hand_angle_listx.append(ComplementConv(int(listofcontent[index][28:30],16)|(int(listofcontent[index][30:32],16)<<8))/100)
                  hand_angle_listy.append(ComplementConv(int(listofcontent[index][32:34],16)|(int(listofcontent[index][34:36],16)<<8))/100)
            elif listofcontent[index][0:4]=="CCCC":
                  foot_acce_x.append(acce_x)
                  foot_acce_y.append(acce_y)
                  foot_acce_z.append(acce_z)
                  foot_gyro_x.append(gyro_x)
                  foot_gyro_y.append(gyro_y)
                  foot_gyro_z.append(gyro_z)
                  foot_angle_listx.append(ComplementConv(int(listofcontent[index][28:30],16)|(int(listofcontent[index][30:32],16)<<8))/100)
                  foot_angle_listy.append(ComplementConv(int(listofcontent[index][32:34],16)|(int(listofcontent[index][34:36],16)<<8))/100)
    
    foot_feature_angle_listx=feature_extraction_func(np.array(foot_angle_listx))
    foot_feature_angle_listx_var=foot_feature_angle_listx[1][1]
    foot_feature_angle_listx_energy=foot_feature_angle_listx[1][2]
    foot_feature_angle_listx_min=foot_feature_angle_listx[1][3]
    foot_feature_angle_listx_max=foot_feature_angle_listx[1][4]
    
    foot_feature_angle_listy=feature_extraction_func(np.array(foot_angle_listy))
    foot_feature_angle_listy_var=foot_feature_angle_listy[1][1]
    foot_feature_angle_listy_energy=foot_feature_angle_listy[1][2]
    foot_feature_angle_listy_min=foot_feature_angle_listy[1][3]
    foot_feature_angle_listy_max=foot_feature_angle_listy[1][4]
    set_of_foot_feature=np.vstack((foot_feature_angle_listx_var,foot_feature_angle_listx_energy,foot_feature_angle_listx_min,foot_feature_angle_listx_max,foot_feature_angle_listy_var,foot_feature_angle_listy_energy,foot_feature_angle_listy_min,foot_feature_angle_listy_max)).T       
    # foot_labels=label*np.ones(foot_feature_angle_listx_var.size)
    clf = joblib.load("./app/controller/model/svmModel.pkl")
    raw_predict_results=clf.predict(set_of_foot_feature)
    padding_results=np.hstack([[raw_predict_results[0]]*2, raw_predict_results, [raw_predict_results[-1]]*2])
    output_results_filter=[]
    # print(raw_predict_results)
    for i in range(raw_predict_results.shape[0]):
        output_results_filter.append(np.round(np.mean(padding_results[i:i+6])))
    output_results_filter = np.array(output_results_filter, dtype=int)
    return output_results_filter
    
    
    
    
    
    
    
    
