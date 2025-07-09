import numpy as np
from seglearn.feature_functions import all_features
from seglearn.transform import FeatureRep
# import matplotlib.pyplot as plt

def feature_extraction_func(data:np.ndarray):
    sample_freq = 20
    window_length = 100
    window_num = int(data.shape[0] / window_length)
    
    data_reshape = data[0:-(data.shape[0] % window_length)].reshape(window_num, window_length)
    
    FeatureTransform = FeatureRep(features=all_features()) #FeatureTransform.features是包含各个特征函数句柄的字典
    feature_list = list(FeatureTransform.features.keys())
    feature_output = np.array(feature_list)[[0, 8, 6, 11, 12, 13, 14]]
    feature_time = []
    for key in feature_output:
        feature_time.append(FeatureTransform.features[key](data_reshape))
    
    data_reshape_freq = (np.abs(np.fft.fft(data_reshape, axis=1)) / window_length)[:, :window_length // 2] #取单边谱
    # plt.figure()
    # plt.plot(data_reshape_freq[0, :])
    # plt.show()
    data_reshape_xaxis = np.fft.fftfreq(window_length) * sample_freq
    feature_freq = []
    for key in feature_output:
        feature_freq.append(FeatureTransform.features[key](data_reshape_freq))
    
    return feature_output, feature_time, feature_freq