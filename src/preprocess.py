'''
数据预处理（插值和平滑）
'''

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import shutil
from scipy.interpolate import UnivariateSpline

dataset_path = 'data_raw'
target_length = 10000
for i in range(11):
    folder_path = dataset_path + f'/{i+1}'
    new_folder_path = f'preprocessed_data/{i+1}'
    os.makedirs(new_folder_path, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            file_path = os.path.join(folder_path, filename)
            data = np.loadtxt(file_path)

            # 滤波
            filtered_data = savgol_filter(data, 51, 3, axis=1)
            # print(filtered_data.shape)
            # 样条插值对齐
            processed_signals = []
            for signal in filtered_data:
                x_origin = np.linspace(0, 1, len(signal))
                x_target = np.linspace(0, 1, target_length)

                spline = UnivariateSpline(x_origin, signal, s=0)
                new_signal = spline(x_target)
                processed_signals.append(new_signal)

            processed_signals = np.array(processed_signals)
            file_path = os.path.join(new_folder_path, filename.replace('.txt', '_filtered.txt'))
            np.savetxt(file_path, processed_signals)

        # plt.figure(figsize=(10, 6))
        # plt.subplot(211)
        # plt.plot(data[0], label='Original')
        # plt.plot(processed_signals[0], label='Filtered')
        # plt.title('Total Acceleration')
        # plt.legend()
        #
        # plt.subplot(212)
        # plt.plot(data[5], label='Original')
        # plt.plot(processed_signals[5], label='Filtered')
        # plt.title('Total Gyro')
        # plt.legend()
        # plt.show()
