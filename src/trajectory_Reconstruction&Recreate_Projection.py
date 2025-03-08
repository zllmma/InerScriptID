'''
生成轨迹投影图片
'''

import os
import numpy as np
import matplotlib.pyplot as plt

new_folder = 'Trajectory_Projection_Diagram' #  创建一个新的文件夹用于存放轨迹投影图片
os.makedirs(new_folder, exist_ok=True)


# 归一化函数 ，用于将数据归一化到指定范围内
def normalize(value, max_value, min_value, size):
    return int((value - min_value)*(size - 1)/(max_value - min_value))


for i in range(11):
    folder = f'preprocessed_data\{i+1}'
    image_folder = os.path.join(new_folder, f'{i+1}')
    os.makedirs(image_folder, exist_ok=True)
    for filename in os.listdir(folder):

        file_path = os.path.join(folder, filename)
        data = np.loadtxt(file_path)
        # print(data)

        acc = np.array(data[:3]).T
        # print(acc)

        gyro = np.array(data[3:]).T

        dt = 1

        # 初始条件
        velocity = np.zeros(3)
        position = np.zeros(3)
        orientation = np.eye(3)  # 初始方向为单位矩阵

        # 轨迹存储
        positions = []

        for i in range(len(acc)):

            current_acc = acc[i]
            # print(current_acc)
            current_gyro = gyro[i]

            # 姿态更新（欧拉角法）
            omega = current_gyro * dt
            theta = np.linalg.norm(omega)

            if theta > 0:
                axis = omega / theta
                cos_theta = np.cos(theta)
                sin_theta = np.sin(theta)

                R = np.array([

                    [cos_theta + axis[0] ** 2 * (1 - cos_theta),
                     axis[0] * axis[1] * (1 - cos_theta) - axis[2] * sin_theta,
                     axis[0] * axis[2] * (1 - cos_theta) + axis[1] * sin_theta],

                    [axis[1] * axis[0] * (1 - cos_theta) + axis[2] * sin_theta,
                     cos_theta + axis[1] ** 2 * (1 - cos_theta),
                     axis[1] * axis[2] * (1 - cos_theta) - axis[0] * sin_theta],

                    [axis[2] * axis[0] * (1 - cos_theta) - axis[1] * sin_theta,
                     axis[2] * axis[1] * (1 - cos_theta) + axis[0] * sin_theta,
                     cos_theta + axis[2] ** 2 * (1 - cos_theta)]
                ])

                orientation = R @ orientation

                acc_world = orientation @ (current_acc - np.array([0, 0, -9.81]))  # 重力去除

                velocity += acc_world * dt
                position += velocity * dt

                positions.append(position.copy())

        positions = np.array(positions)

        # # 可视化轨迹
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
        # ax.set_xlabel('X Position (m)')
        # ax.set_ylabel('Y Position (m)')
        # ax.set_zlabel('Z Position (m)')
        # ax.set_title('3D Trajectory')
        # plt.show()

        # 绘制组合投影图
        x = positions[:, 0]
        y = positions[:, 1]
        z = positions[:, 2]

        size = 224
        xy_image = np.zeros((size, size, 3), dtype=np.uint8)
        xz_image = np.zeros((size, size, 3), dtype=np.uint8)
        yz_image = np.zeros((size, size, 3), dtype=np.uint8)

        max_x = np.max(x)
        min_x = np.min(x)
        max_y = np.max(y)
        min_y = np.min(y)
        max_z = np.max(z)
        min_z = np.min(z)

        # 绘制XY平面
        for i in range(len(x)):
            x_pix = normalize(x[i], max_x, min_x, size)
            y_pix = normalize(y[i], max_y, min_y, size)
            xy_image[size - 1 - y_pix, x_pix] = [255, 0, 0]  # 红色

        # 绘制XZ平面
        for i in range(len(x)):
            x_pix = normalize(x[i], max_x, min_x, size)
            z_pix = normalize(z[i], max_z, min_z, size)
            xz_image[size - 1 - z_pix, x_pix] = [0, 255, 0]  # 绿色

        # 绘制YZ平面
        for i in range(len(y)):
            y_pix = normalize(y[i], max_y, min_y, size)
            z_pix = normalize(z[i], max_z, min_z, size)
            yz_image[size - 1 - z_pix, y_pix] = [0, 0, 255]  # 蓝色

        # 合并三个通道图像
        combined_image = np.zeros((size, size, 3), dtype=np.uint8)
        combined_image[:, :, 0] = xy_image[:, :, 0]  # 红色通道
        combined_image[:, :, 1] = xz_image[:, :, 1]  # 绿色通道
        combined_image[:, :, 2] = yz_image[:, :, 2]  # 蓝色通道

        # 显示合并后的图像
        plt.imshow(combined_image)
        plt.title('')
        plt.xlabel('')
        plt.ylabel('')
        plt.axis('off')  # 不显示坐标轴

        name = filename.split('_')[0] + '.png'
        image_path = os.path.join(image_folder, name)
        print(image_path)
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0)
        plt.clf()
