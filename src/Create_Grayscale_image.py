'''
生成信号灰度图片
'''

import os
import numpy as np
import matplotlib.pyplot as plt


def normalize(signal):
    return (signal - np.min(signal)) / (np.max(signal) - np.min(signal)) * 255


folder = 'data_raw'
new_folder = 'Grayscale_images_mixed'
os.makedirs(new_folder, exist_ok=True)

for i in range(11):
    current_folder = os.path.join(folder, f'{i+1}')
    current_new_folder = os.path.join(new_folder, f'{i+1}')
    os.makedirs(current_new_folder, exist_ok=True)
    for filename in os.listdir(current_folder):
        data_path = os.path.join(current_folder, filename)
        data = np.loadtxt(data_path)
        # print(data.shape)
        # name = filename.split('_')[0]
        # image_folder = os.path.join(current_new_folder, name)
        # os.makedirs(image_folder, exist_ok=True)

        gray_image_matrix = np.zeros((100, 100))
        for j in range(6):
            current_signal = data[j]
            current_signal = current_signal.reshape(100, 100)
            gray_image_matrix += normalize(current_signal).astype(np.uint8)
        gray_image_matrix /= 6

        name = filename.split('_')[0] + '.png'
        image_path = os.path.join(current_new_folder, name)
        print(image_path)
        plt.imshow(gray_image_matrix, cmap='gray')
        plt.title('')
        plt.xlabel('')
        plt.ylabel('')
        plt.axis('off')  # 不显示坐标轴
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.clf()
