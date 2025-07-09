import numpy as np
import math
import sklearn.svm as svm
from PyEMD import EMD
from sklearn import neighbors
import scipy.io as scio
# import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft
import joblib


###################################数据加载###################################
class data_load_process:
#手部训练数据加载 读取位置数据存入矩阵
#输入：手部训练数据位置 手部训练标签位置
#输出：手部训练数据矩阵 手部训练标签矩阵
#     def train_hand_data_loader(train_hand_data_path,train_hand_label_path):
#         input_data = scio.loadmat(train_hand_data_path)
#         input_label = scio.loadmat(train_hand_label_path)
#         train_hand_data = input_data['tianke32']
#         train_hand_label_temp = input_label['lable_tianke32']
#         train_hand_label = train_hand_label_temp.reshape(train_hand_label_temp.shape[1],)
#         #train_hand_label = (train_hand_label.astype(np.float64)).T
#         feature_mean_data = []
#         feature_var_data = []
#         feature_sum_data = []   
#         Num_data = train_hand_data.shape[1]
#         for i in range(Num_data):
#             slide_data = train_hand_data[:, i]
#             mean_slide_data = np.mean(slide_data)
#             var_slide_data = np.var(slide_data)
#             sum_slide_data = np.sum(abs(slide_data))
#             feature_mean_data.append(mean_slide_data)
#             feature_var_data.append(var_slide_data)
#             feature_sum_data.append(sum_slide_data)
#         train_feature = [feature_mean_data, feature_var_data, feature_sum_data]
#         train_hand_feature_array = np.array(train_feature).T
#         return (train_hand_feature_array,train_hand_label)
    
# #脚部训练数据加载 读取位置数据存入矩阵
# #输入：脚部训练数据位置 脚部训练标签位置
# #输出：脚部训练数据矩阵 脚部训练标签矩阵
#     def train_foot_data_loader(train_foot_data_path,train_foot_label):       
#         train_data = np.loadtxt(train_foot_data_path, delimiter=',')
#         train_label = np.loadtxt(train_foot_label_path, delimiter=',')        
#         feature_mean_data = []
#         feature_var_data = []
#         feature_sum_data = []        
#         Num_data = train_data.shape[1]
#         for i in range(Num_data):
#             slide_data = train_data[:, i]
#             mean_slide_data = np.mean(slide_data)
#             var_slide_data = np.var(slide_data)
#             sum_slide_data = np.sum(abs(slide_data))
#             feature_mean_data.append(mean_slide_data)
#             feature_var_data.append(var_slide_data)
#             feature_sum_data.append(sum_slide_data)
#         train_feature = [feature_mean_data, feature_var_data, feature_sum_data]
#         train_feature_array = np.array(train_feature).T
#         return (train_feature_array,train_label)
    
#测试数据加载 读取位置数据 按频率降采样 存入矩阵
#输入：测试数据位置 测试数据采样频率
#输出：降采样后的 手部加速度数据矩阵Acce0 手部角速度矩阵 脚部加速度数据矩阵Acce1 脚部角速度矩阵    
    def test_data_loader(test_data,test_data_freq):
        
        #test_data = np.loadtxt(test_filename, delimiter=',', skiprows=1)
        #time_stamp = np.arange(1, test_data.shape[0] + 1) / test_data_freq
        train_sample_rate = 20
        down_sample_rate = math.floor(test_data_freq/train_sample_rate)
        
        Ori_Acce0 = np.array(test_data)[:, 1:4]
        Ori_Gyro0 = np.array(test_data)[:, 4:7]
        Ori_Acce1 = np.array(test_data)[:, 10:13]
        Ori_Gyro1 = np.array(test_data)[:, 13:16]
        ds_Acce0 = Ori_Acce0[np.arange(0, test_data.shape[0], down_sample_rate)]
        ds_Gyro0 = Ori_Gyro0[np.arange(0, test_data.shape[0], down_sample_rate)]
        ds_Acce1 = Ori_Acce1[np.arange(0, test_data.shape[0], down_sample_rate)]
        ds_Gyro1 = Ori_Gyro1[np.arange(0, test_data.shape[0], down_sample_rate)]
        #LWH_counter should be in loop, can be 0-2
        
        return (ds_Acce0, ds_Gyro0, ds_Acce1, ds_Gyro1)
    
##################################数据加载结束################################## 

##################################三种模型训练##################################   
class Model_generate:
#特征提取函数
#输入： 训练数据矩阵
#输出： 训练特征矩阵
    def train_feature_extract(train_data):
        col = train_data.shape[1]
        mean_data = []
        var_data = []
        sum_data = []
        for i in range(col):
            slide_data = train_data[:, i]
            mean_slide_data = np.mean(slide_data)
            var_slide_data = np.var(slide_data)
            sum_slide_data = np.sum(slide_data)
            mean_data.append(mean_slide_data)
            var_data.append(var_slide_data)
            sum_data.append(sum_slide_data)
        train_feature = [mean_data, var_data, sum_data]
        train_feature_array = np.array(train_feature).T
        return (train_feature_array)
    
#SVM模型训练
#输入： 训练特征矩阵 训练标签矩阵
#输出： SVM模型  
    def train_svm_model(train_feature_array,train_label):
        
        clf = svm.SVC(kernel='linear')
        #svm_Model = 
        clf.fit(train_feature_array, train_label)
        return(clf)
    
#KNN模型训练
#输入： 训练特征矩阵 训练标签矩阵
#输出： KNN模型     
    def train_k1nn_model(train_feature_array, train_label, k=1):
  
        k1nn = neighbors.KNeighborsClassifier(k)
        #k1nn_model = 
        k1nn.fit(train_feature_array, train_label)
        return(k1nn)
    
#KNN模型训练
#输入： 训练特征矩阵 训练标签矩阵
#输出： KNN模型     
    def train_k3nn_model(train_feature_array, train_label, k = 7):
        k3nn = neighbors.KNeighborsClassifier(k)
        #k3nn_model = 
        k3nn.fit(train_feature_array, train_label)
        return(k3nn)
    
##################################数据预处理################################## 
class EMD_process:
#EMD滤波
#输入： 一维数据矩阵
#输出： EMD降噪数据矩阵    
    def EMD_model(ori_data):
        emd = EMD() #创建EMD对象
        Out_put_data = []
        for number_counter in range(ori_data.shape[1]):
            input_signal = ori_data[:,number_counter]
            IMFs = emd.emd(input_signal, max_imf=9).T
            #(n1, md) = IMFs.shape
            IMF24 = np.sum(IMFs[:,1:4], axis=1)
            #plt.plot(IMF24)
            Out_put_data.append(IMF24)
        EMD_data = np.array(Out_put_data).T
        # plt.plot(EMD_data[:,0])
        # plt.show()
        return(EMD_data)
    
#################################数据预处理结束#################################   

################################测试数据分段分类################################# 
class classifier:
#测试数据特征提取
#输入： EMD降噪数据矩阵
#输出： 降噪数据特征矩阵（分段）
    def feature_extract(EMD_data):     
        test_data = []
        test_record = np.arange(1, int(EMD_data.shape[0]/100)) * 100
        for i in range(test_record.shape[0]):
            test_data.append(EMD_data[np.arange(test_record[i], test_record[i]+100)])  
            
        segm_data = np.array(test_data)
        test_feature_array = []
        for number_counter in range(segm_data.shape[2]):
            mean_segm_data = []
            var_segm_data = []
            sum_segm_data = [] 
            one_line_data = segm_data[:,:,number_counter]
            for i in range(len(one_line_data)):
                slide_data = one_line_data[i]
                mean_segm_data.append(np.mean(slide_data))
                var_segm_data.append(np.var(slide_data))
                sum_segm_data.append(np.sum(abs(slide_data)))
            one_line_feature = [mean_segm_data, var_segm_data, sum_segm_data]
            test_feature_array.append(one_line_feature)
        feature_array = np.array(test_feature_array).T
        return(feature_array)

#SVM模型分类
#输入： SVM模型 降噪数据特征矩阵（分段）
#输出： 降噪数据分类结果（分段）     
    def svm_classify(clf, test_feature_array):            
        svm_input = np.array(test_feature_array)
        label_output = []
        for i in range(svm_input.shape[0]):
            one_input = svm_input[i,:,:].T          
            svmResult = clf.predict(np.array(one_input))
            label_output.append(svmResult)
        output_label = np.array(label_output)
        return(output_label)

#KNN模型分类
#输入： KNN模型 降噪数据特征矩阵（分段）
#输出： 降噪数据分类结果（分段）   
    def k1nn_classify(k1nn, test_feature_array): 
        k1nn_input = np.array(test_feature_array)
        label_output = []
        for i in range(k1nn_input.shape[0]):
            one_input = k1nn_input[i,:,:].T          
            k1nnResult = k1nn.predict(np.array(one_input))
            label_output.append(k1nnResult)
        output_label = np.array(label_output)
        return(output_label)
    
#KNN模型分类
#输入： KNN模型 降噪数据特征矩阵（分段）
#输出： 降噪数据分类结果（分段）    
    def k3nn_classify(k3nn, test_feature_array): 
        k3nn_input = np.array(test_feature_array)
        label_output = []
        for i in range(k3nn_input.shape[0]):
            one_input = k3nn_input[i,:,:].T          
            k3nnResult = k3nn.predict(np.array(one_input))
            label_output.append(k3nnResult)
        output_label = np.array(label_output)
        return(output_label)

################################打分（很有道理）################################   
class reasonable_3_to_1:
    def fusion(SVM_label, k1nn_label, k3nn_label): 
        fusion_result = []
        for loop_counter in range(SVM_label.shape[1]):
            for inner_counter in range(SVM_label.shape[0]):
                if inner_counter == 0 or inner_counter == SVM_label.shape[0] - 1:
                    pass
                else:
                    if SVM_label[inner_counter - 1, loop_counter] == 1 and SVM_label[inner_counter + 1, loop_counter] == 1 and SVM_label[inner_counter, loop_counter] == 0:
                        SVM_label[inner_counter, loop_counter] = 1
                    if k1nn_label[inner_counter - 1, loop_counter] == 1 and k1nn_label[inner_counter + 1, loop_counter] == 1 and k1nn_label[inner_counter, loop_counter] == 0:
                        k1nn_label[inner_counter, loop_counter] = 1
                    if k3nn_label[inner_counter - 1, loop_counter] == 1 and k3nn_label[inner_counter + 1, loop_counter] == 1 and k3nn_label[inner_counter, loop_counter] == 0:
                        k3nn_label[inner_counter, loop_counter] = 1
                
        for loop_counter in range(SVM_label.shape[1]):
            one_result = SVM_label[:,loop_counter] + k1nn_label[:,loop_counter] + k3nn_label[:,loop_counter]
            index = (one_result == 2)
            one_result[index] = 3
            index = (one_result == 1)
            one_result[index] = 0
            fusion_result.append(one_result)
        return (fusion_result)    
#################################打分结束（有理）#################################       

#################################区间参数计算#################################      
class varbel_calculation:
#峰值点个数计算 大聪明写的
#输入： 信号 阈值
#输出： 点数量 点坐标
    def peak_amp(block_wave,threshhold):
        peak_index = []
        peak_data = []
        num=1;
        data = block_wave
        for r_counter in range(len(block_wave) - 1):
            if r_counter != 0:
                if (data[r_counter] > data[r_counter - 1]) and ((data[r_counter] > data[r_counter + 1] or (data[r_counter] == data[r_counter + 1]))):
                    if(data[r_counter] > threshhold):
                        peak_index.append(r_counter)
                        peak_data.append(data[r_counter])
                        num = num + 1;
        return(peak_index,peak_data)

#数据获取
#输入： 分类结果（有道理的） EMD滤波数据
#输出： 目标特征        
    def hand_verbal_gets(fusion_result,EMD_Acce0): 
        fusion_label = np.array(fusion_result)
#基本参数（可活）
        block_len = 100;
        resonable_threshhold = 1
        Freq = 20
        varbel_list = []
#根据标签找到异常区间（容错 = 1）
        for loop_counter in range(fusion_label.shape[0]):
            one_label = fusion_label[loop_counter,:]
            one_data = EMD_Acce0[:,loop_counter]
            start_flag = 0;
            start_temp = [];
            start_list = [];
            end_list = [];
            for label_counter in range(one_label.shape[0]):
                if start_flag == 0 and one_label[label_counter] != 0:
                    start_flag = 1
                    start_temp = label_counter                    
                if start_flag == 1 and one_label[label_counter] == 0:
                    start_flag = 0
                    if label_counter - start_temp - 1 > 2:
                        start_list.append(start_temp)
                        end_list.append(label_counter)
                if label_counter == (one_label.shape[0] - 1) and start_flag ==1:
                    start_flag = 0
                    if label_counter - start_temp - 1 > 2:
                        start_list.append(start_temp)
                        end_list.append(label_counter)
#异常区间get                        
            start_list = np.array(start_list)
            end_list = np.array(end_list)
#计算参数
            for list_counter in range(len(start_list)):
#开始 结束 时长
                start_time = start_list[list_counter] * block_len / Freq
                end_time = end_list[list_counter] * block_len / Freq
                total_time = end_time - start_time;
#均值 方差
                block_wave = one_data[block_len * (start_list[list_counter]) : block_len * (end_list[list_counter])]
                mean_block_wave = np.mean(block_wave)
                Var_block_wave = np.var(block_wave)
#峰值点 坐标
                [peak_index_neg,peak_data_neg] = varbel_calculation.peak_amp( - block_wave, - mean_block_wave + resonable_threshhold)
                [peak_index_pos,peak_data_pos] = varbel_calculation.peak_amp( block_wave, mean_block_wave + resonable_threshhold)
#点数 幅值 速度
                if (len(peak_index_neg) + len(peak_index_pos)) != 0 :
                    peak_num = (len(peak_index_neg) + len(peak_index_pos))/ 2
                    peak_apm = ((sum(peak_data_neg) + sum(peak_data_pos))) / (len(peak_index_neg) + len(peak_index_pos) )
                    speed =  (1/2) * peak_apm * ((block_len / Freq) / peak_num ) ** 2
                else:
                    peak_num = 0
                    peak_apm = 0
                    speed = 0
#傅里叶变换
#功率谱 频率
                # yy=fft(block_wave)                     #快速傅里叶变换
                # yreal = yy.real               # 获取实数部分
                # yimag = yy.imag               # 获取虚数部分  
                #yf=abs(fft(block_wave))                # 取模
                yf1=abs(fft(block_wave))/((len(block_wave)/2))           #归一化处理
                yf2 = yf1[range(int(len(block_wave)/2))]  #由于对称性，只取一半区间
                xf = np.arange(len(block_wave))        # 频率
                #xf1 = xf
                xf2 = xf[range(int(len(block_wave)/2))]
                index = (yf2 == max(yf2))              
                # FFT_block_wave = fft(block_wave)  
                # p = FFT_block_wave * np.conj(FFT_block_wave) / len(FFT_block_wave)
                # max_index= np.where(p == p.max() ).

                freq_max = Freq * xf2[index] /len(block_wave)
                freq = freq_max[0]
                varbel_list.append([loop_counter, start_time, end_time, total_time, abs(mean_block_wave), Var_block_wave, peak_num, peak_apm, speed, freq])
        varbel_list = np.array(varbel_list)
        return(varbel_list)

    def foot_verbal_gets(fusion_result,EMD_Acce1): 
       fusion_label = np.array(fusion_result)
       block_len = 100;
       resonable_threshhold = 1
       Freq = 20
       varbel_list = []
       for loop_counter in range(fusion_label.shape[0]):
           one_label = fusion_label[loop_counter,:]
           one_data = EMD_Acce1[:,loop_counter]
           start_flag = 0;
           start_temp = [];
           start_list = [];
           end_list = [];
           for label_counter in range(one_label.shape[0]):
               if start_flag == 0 and one_label[label_counter] != 0:
                   start_flag = 1
                   start_temp = label_counter
                   
               if start_flag == 1 and one_label[label_counter] == 0:
                   start_flag = 0
                   if label_counter - start_temp - 1 > 2:
                       start_list.append(start_temp)
                       end_list.append(label_counter)
               if label_counter == (one_label.shape[0] - 1) and start_flag ==1:
                   start_flag = 0
                   if label_counter - start_temp - 1 > 2:
                       start_list.append(start_temp)
                       end_list.append(label_counter)
                       
           start_list = np.array(start_list)
           end_list = np.array(end_list)
           for list_counter in range(len(start_list)):
               start_time = start_list[list_counter] * block_len / Freq
               end_time = end_list[list_counter] * block_len / Freq
               total_time = end_time - start_time;
               block_wave = one_data[block_len * (start_list[list_counter]) : block_len * (end_list[list_counter])]
               mean_block_wave = np.mean(block_wave)
               Var_block_wave = np.var(block_wave)
               [peak_index_neg,peak_data_neg] = varbel_calculation.peak_amp( - block_wave, - mean_block_wave + resonable_threshhold)
               [peak_index_pos,peak_data_pos] = varbel_calculation.peak_amp( block_wave, mean_block_wave + resonable_threshhold)
               if (len(peak_index_neg) + len(peak_index_pos)) != 0 :
                   peak_num = (len(peak_index_neg) + len(peak_index_pos))/ 2
                   peak_apm = ((sum(peak_data_neg) + sum(peak_data_pos))) / (len(peak_index_neg) + len(peak_index_pos) )    
                   
               else:
                   peak_num = 0
                   peak_apm = 0
#傅里叶变换
#功率谱 频率 速度                 
               # yy=fft(block_wave)                     #快速傅里叶变换
               # yreal = yy.real               # 获取实数部分
               # yimag = yy.imag               # 获取虚数部分  
               #yf=abs(fft(block_wave))                # 取模
               yf1=abs(fft(block_wave))/((len(block_wave)/2))           #归一化处理
               yf2 = yf1[range(int(len(block_wave)/2))]  #由于对称性，只取一半区间
               xf = np.arange(len(block_wave))        # 频率
               #xf1 = xf
               xf2 = xf[range(int(len(block_wave)/2))]
               index = (yf2 == max(yf2))              
               # FFT_block_wave = fft(block_wave)  
               # p = FFT_block_wave * np.conj(FFT_block_wave) / len(FFT_block_wave)
               # max_index= np.where(p == p.max() )
               freq_max = Freq * xf2[index] /len(block_wave)
               freq = freq_max[0]
               speed = (Freq * freq) * (1/2) * 9.8 *  (peak_apm) * ((1/Freq) ** 2)
               #speed代表的步幅（米/步），freq代表步频（步/s），speed*freq 代表步速（米/s）
               varbel_list.append([loop_counter, start_time, end_time, total_time, abs(mean_block_wave), Var_block_wave, peak_num, peak_apm, speed, freq])
       varbel_list = np.array(varbel_list)
       return(varbel_list)
#################################区间参数计算结束#################################  

# #输出好看
np.set_printoptions(suppress=True)  
test_data_freq = 200 
# #输入文件位置     
# #test_filename = 'C:/Users/AA/Downloads/svm-test/test123.csv'
# test_filename = 'C:/Users/Administrator/Desktop/svm-test-天可/Mpu_Data_2020_08_11_10_28_33.csv'
# train_hand_data_path = 'C:/Users/Administrator/Desktop/svm-test-天可/class_data_hand.mat'
# train_hand_label_path = 'C:/Users/Administrator/Desktop/svm-test-天可/class_label_hand.mat'

# train_foot_data_path = 'C:/Users/Administrator/Desktop/svm-test-天可/train_data.csv'
# train_foot_label_path = 'C:/Users/Administrator/Desktop/svm-test-天可/train_label.csv'
# #输入文件频率
# test_data_freq = 200 

# #训练数据特征提取
# train_hand_feature_array, train_hand_label = data_load_process.train_hand_data_loader(train_hand_data_path, train_hand_label_path)
# train_foot_feature_array, train_foot_label = data_load_process.train_foot_data_loader(train_foot_data_path, train_foot_label_path)        

# #训练模型
# SVM_hand_model = Model_generate.train_svm_model(train_hand_feature_array,train_hand_label)  
# k1nn_hand_model = Model_generate.train_k1nn_model(train_hand_feature_array,train_hand_label)  
# k3nn_hand_model = Model_generate.train_k3nn_model(train_hand_feature_array,train_hand_label)  

# SVM_foot_model = Model_generate.train_svm_model(train_foot_feature_array,train_foot_label)  
# k1nn_foot_model = Model_generate.train_k1nn_model(train_foot_feature_array,train_foot_label)  
# k3nn_foot_model = Model_generate.train_k3nn_model(train_foot_feature_array,train_foot_label)  

# joblib.dump((SVM_hand_model,k1nn_hand_model,k3nn_hand_model),'hand.dpl')

# joblib.dump((SVM_foot_model,k1nn_foot_model,k3nn_foot_model),'foot.dpl')

###############以下是离线模型#########
SVM_hand_model, k1nn_hand_model, k3nn_hand_model = joblib.load('./app/controller/model/hand.dpl')
SVM_foot_model, k1nn_foot_model, k3nn_foot_model= joblib.load('./app/controller/model/foot.dpl')

def outputfeature_svm_knn(test_data):
    ds_Acce0, ds_Gyro0, ds_Acce1, ds_Gyro1 = data_load_process.test_data_loader(test_data, test_data_freq)   
    raw_output=[ds_Acce0, ds_Gyro0, ds_Acce1, ds_Gyro1]

    #测试数据滤波
    EMD_Acce0 = EMD_process.EMD_model(ds_Acce0)
    EMD_Acce1 = EMD_process.EMD_model(ds_Acce1)
    
    #测试数据特征提取
    feature_Acce0 = classifier.feature_extract(EMD_Acce0)
    feature_Acce1 = classifier.feature_extract(EMD_Acce1)
            # plt.plot(EMD_Acce1[:,2])
            # plt.show()
    
    #测试数据分类
    SVM_hand_pred_label = classifier.svm_classify(SVM_hand_model,feature_Acce0)
    K1nn_hand_pred_label = classifier.k1nn_classify(k1nn_hand_model,feature_Acce0)
    k3nn_hand_pred_label = classifier.k3nn_classify(k3nn_hand_model,feature_Acce0)
    
    SVM_foot_pred_label = classifier.svm_classify(SVM_foot_model,feature_Acce1)
    K1nn_foot_pred_label = classifier.k1nn_classify(k1nn_foot_model,feature_Acce1)
    k3nn_foot_pred_label = classifier.k3nn_classify(k3nn_foot_model,feature_Acce1)
    
    #分类结果投票（有道理）
    hand_fusion_result = reasonable_3_to_1.fusion(SVM_hand_pred_label, K1nn_hand_pred_label, k3nn_hand_pred_label)
    foot_fusion_result = reasonable_3_to_1.fusion(SVM_foot_pred_label, K1nn_foot_pred_label, k3nn_foot_pred_label)
    
    #病症区间参数计算
    hand_varbel_list = varbel_calculation.hand_verbal_gets(hand_fusion_result,EMD_Acce0)
    foot_varbel_list = varbel_calculation.foot_verbal_gets(foot_fusion_result,EMD_Acce1)
    return hand_varbel_list, foot_varbel_list, raw_output
