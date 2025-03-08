"""
将 data_raw 目录下的所有 txt 文件重命名为 csv 文件
"""
import os

for root, dirs, files in os.walk('.'):
    for filename in files:
        if filename.lower().endswith('.txt'):
            old_path = os.path.join(root, filename)
            new_filename = os.path.splitext(filename)[0] + '.csv'
            new_path = os.path.join(root, new_filename)
            
            if not os.path.exists(new_path):
                os.rename(old_path, new_path)
                print(f'已重命名: {old_path} -> {new_path}')
            else:
                print(f'目标文件已存在，跳过: {old_path}')