import numpy as np
from PyEMD import EMD

# import sklearn.svm as svm
# from sklearn import neighbors
import joblib
import math

# import matplotlib.pyplot as plt
# import scipy.io as scio
from scipy.fftpack import fft


class data_load_process:
    def test_data_loader(test_data,test_data_freq):
        
        # test_data = np.loadtxt(test_filename, delimiter=',', skiprows=1)
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
 
class EMD_process:
    
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
  
class classifier:
    
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
    
    def svm_classify(clf, test_feature_array):            
        svm_input = np.array(test_feature_array)
        label_output = []
        for i in range(svm_input.shape[0]):
            one_input = svm_input[i,:,:].T          
            svmResult = clf.predict(np.array(one_input))
            label_output.append(svmResult)
        output_label = np.array(label_output)
        return(output_label)
    
    def k1nn_classify(k1nn, test_feature_array): 
        k1nn_input = np.array(test_feature_array)
        label_output = []
        for i in range(k1nn_input.shape[0]):
            one_input = k1nn_input[i,:,:].T          
            k1nnResult = k1nn.predict(np.array(one_input))
            label_output.append(k1nnResult)
        output_label = np.array(label_output)
        return(output_label)
    
    def k3nn_classify(k3nn, test_feature_array): 
        k3nn_input = np.array(test_feature_array)
        label_output = []
        for i in range(k3nn_input.shape[0]):
            one_input = k3nn_input[i,:,:].T          
            k3nnResult = k3nn.predict(np.array(one_input))
            label_output.append(k3nnResult)
        output_label = np.array(label_output)
        return(output_label)
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
    
class varbel_calculation:
    
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
        
    def hand_verbal_gets(fusion_result,EMD_Acce0): 
        fusion_label = np.array(fusion_result)
        block_len = 100;
        resonable_threshhold = 1
        Freq = 20
        varbel_list = []
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
                varbel_list.append([loop_counter, start_time, end_time, total_time, abs(mean_block_wave), Var_block_wave, peak_num, peak_apm, freq])
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
               varbel_list.append([loop_counter, start_time, end_time, total_time, abs(mean_block_wave), Var_block_wave, peak_num, peak_apm, freq])
       varbel_list = np.array(varbel_list)
       return(varbel_list)



np.set_printoptions(suppress=True)       
test_data_freq = 200 




###############以下是离线模型#########
SVM_hand_model, k1nn_hand_model, k3nn_hand_model = joblib.load('./app/controller/model/hand.dpl')
SVM_foot_model, k1nn_foot_model, k3nn_foot_model= joblib.load('./app/controller/model/foot.dpl')



def outputfeature_svm_knn(test_data):
    ds_Acce0, ds_Gyro0, ds_Acce1, ds_Gyro1 = data_load_process.test_data_loader(test_data, test_data_freq)   
    raw_output=[ds_Acce0, ds_Gyro0, ds_Acce1, ds_Gyro1]

    EMD_Acce0 = EMD_process.EMD_model(ds_Acce0)
    EMD_Acce1 = EMD_process.EMD_model(ds_Acce1)
    
    feature_Acce0 = classifier.feature_extract(EMD_Acce0)
    feature_Acce1 = classifier.feature_extract(EMD_Acce1)
    
    SVM_hand_pred_label = classifier.svm_classify(SVM_hand_model,feature_Acce0)
    K1nn_hand_pred_label = classifier.k1nn_classify(k1nn_hand_model,feature_Acce0)
    k3nn_hand_pred_label = classifier.k3nn_classify(k3nn_hand_model,feature_Acce0)
    
    SVM_foot_pred_label = classifier.svm_classify(SVM_foot_model,feature_Acce1)
    K1nn_foot_pred_label = classifier.k1nn_classify(k1nn_foot_model,feature_Acce1)
    k3nn_foot_pred_label = classifier.k3nn_classify(k3nn_foot_model,feature_Acce1)
     
    
    hand_fusion_result = reasonable_3_to_1.fusion(SVM_hand_pred_label, K1nn_hand_pred_label, k3nn_hand_pred_label)
    foot_fusion_result = reasonable_3_to_1.fusion(SVM_foot_pred_label, K1nn_foot_pred_label, k3nn_foot_pred_label)
    
    
    hand_varbel_list = varbel_calculation.hand_verbal_gets(hand_fusion_result,EMD_Acce0)
    foot_varbel_list = varbel_calculation.foot_verbal_gets(foot_fusion_result,EMD_Acce1)
    return hand_varbel_list, foot_varbel_list, raw_output