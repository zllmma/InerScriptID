'''
提取统计特征（时域、频域）
'''

import os

import pandas as pd
import numpy as np
from scipy.fft import fft
from scipy.signal import welch


def extract_time_domain_features(signal):
    # 时域特征
    mean = np.mean(signal)
    std = np.std(signal)
    max_val = np.max(signal)
    min_val = np.min(signal)
    rms = np.sqrt(np.mean(signal**2))
    return [mean, std, max_val, min_val, rms]


def extract_frequency_domain_features(signal, sampling_rate):
    # 快速傅里叶变换
    N = len(signal)
    fft_vals = fft(signal)
    freqs = np.fft.fftfreq(N, 1 / sampling_rate)

    # 取正频部分
    positive_freqs = freqs[:N // 2]
    positive_fft_vals = np.abs(fft_vals[:N // 2])

    # 频域特征
    peak_freq = positive_freqs[np.argmax(positive_fft_vals)]  # 峰值频率
    power_spectrum = np.sum(positive_fft_vals**2)  # 频谱能量

    return [peak_freq, power_spectrum]


sampling_rate = 500  # 500 Hz

for i in range(11):
    folder_path = f'preprocessed_data/{i+1}'
    new_folder_path = f'statistic_feature/{i+1}'
    os.makedirs(new_folder_path, exist_ok=True)

    for filename in os.listdir(folder_path):
        combined_features = []
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            data = np.loadtxt(file_path)

            for j in range(len(data)):
                time_domain_feature = extract_time_domain_features(data[j])
                frequency_domain_feature = extract_frequency_domain_features(data[j], sampling_rate)
                combined_feature = time_domain_feature + frequency_domain_feature
                combined_features.append(combined_feature)

            combined_features_array = np.array(combined_features)
            file_path = os.path.join(new_folder_path, filename.replace('_filtered.txt', 'statistic_feature.txt'))
            np.savetxt(file_path, combined_features_array)
