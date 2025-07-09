import numpy as np
import pywt
def tremor_detection(data):
    frequency  = 20
    block_len = 10
    
    duration_list = []
    datalen = data[0].size
    durationlen = int(datalen*(1/frequency)/60)
    
    for LWH_counter in range(1,4):
        x = data[LWH_counter-1]
        x = x[0:durationlen*frequency*60]
        print(x)
        if len(x) == 0:
            return [], [], []
        db4 = pywt.Wavelet('db4')
        cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = pywt.wavedec(x, db4, level=7)
        coeffs = [cA7*0, cD7, cD6, cD5, cD4*0, cD3*0, cD2*0, cD1*0]
        Signal = pywt.waverec(coeffs, db4)
        # plt.plot(x_out)
        # plt.show()
        # pass
        Var_Sig =np.var(Signal)
        # Mean_Sig = np.mean(Signal)
        search_mark = []
        # listltcounter=[]
        for LT_counter in range(1, int(Signal.size/block_len)+1):
            # listltcounter.append(LT_counter)
            if np.var(Signal[block_len * (LT_counter - 1) : block_len * (LT_counter)]) > Var_Sig / 5:
                search_mark.append(LT_counter)
        search_len = len(search_mark)
        search_flag = 0
        start_list = []
        end_list = []
        for search_counter in range(1, search_len+1):
            if search_flag==0:
                search_flag=1
            else:
                if search_mark[search_counter-1]-search_mark[search_counter-2]==1:
                    search_flag=search_flag+1
                else:
                    if search_flag > frequency / block_len * 2:
                        start_list.append(search_mark[search_counter-search_flag-1])
                        end_list.append(search_mark[search_counter-2])
                    search_flag=0
        duration = set()
        for find_counter in range(1,len(start_list)+1):
            block_wave = Signal[block_len * (start_list[find_counter-1]-1):block_len * (end_list[find_counter-1]-1)]
            y_f = np.fft.fft(block_wave)
            # Druation = block_wave.size/frequency
            Sampling_points = block_wave.size
            f_s = frequency
            f_x=np.arange(0,f_s+f_s/(Sampling_points -1),f_s/(Sampling_points -1))
            # t2=f_x-f_s/2
            # shift_f = np.abs(np.fft.fftshift(y_f))
            freq = np.abs(y_f[0:int(f_x.size/2)-1])
            q = np.where(freq==np.max(freq[1:freq.size-1]))
            # tumor_counter = 0
            if q[0][0] * f_s/(Sampling_points -1) > 0.2:
                start_time = start_list[find_counter-1] *  block_len / frequency
                end_time = end_list[find_counter-1] *  block_len / frequency
                duration.update(set(np.arange(start_time,end_time+0.5,0.5)))
                # print("发生手部震颤，时段"+str(start_time)+'-'+str(end_time))
                # tumor_counter = 1
            # if tumor_counter == 0:
                # print('该时段未发生震颤')
        
        duration_list.append(duration)
    
    duration_intersect = list(duration_list[0].intersection(duration_list[1]).intersection(duration_list[2]))
    duration_intersect.sort()
    duration_union = list(duration_list[0].intersection(duration_list[1]).union(duration_list[0].intersection(duration_list[2])).union(duration_list[1].intersection(duration_list[2])))
    duration_union.sort()

    tramor_intersect_output = [0] * durationlen
    for i in range(0,len(duration_intersect)):
        num = int(duration_intersect[i]/60)
        tramor_intersect_output[num] = tramor_intersect_output[num] + 1
        
    tramor_union_output = [0] * durationlen
    for i in range(0,len(duration_union)):
        num = int(duration_union[i]/60)
        tramor_union_output[num] = tramor_union_output[num] + 1
    tramor_level = np.zeros(len(tramor_union_output))
    for i in np.where(np.array(tramor_union_output) < 9):
        tramor_level[i] = 0
    for i in np.intersect1d(np.where(np.array(tramor_union_output) >=9), np.where(np.array(tramor_union_output) < 46)):
        tramor_level[i] = 1
    for i in np.intersect1d(np.where(np.array(tramor_union_output) >=46), np.where(np.array(tramor_union_output) < 83)):
        tramor_level[i] = 2
    for i in np.where(np.array(tramor_union_output) >=84):
        tramor_level[i] = 3
    return tramor_intersect_output, tramor_union_output, tramor_level.astype(int).tolist()