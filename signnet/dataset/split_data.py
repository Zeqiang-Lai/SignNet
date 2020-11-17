import os
import numpy as np
import shutil



def move_file(srcfile, dstpath):
    #  判断移动文件是否存在        
    if os.path.isfile(srcfile):
        #  分离文件名和路径
        fpath, fname = os.path.split(srcfile)             
        #  创建路径
        if not os.path.exists(dstpath):
            os.makedirs(dstpath)
        #  移动文件                   
        shutil.copy(srcfile, dstpath + '/' + fname)          


def split_data(data, ratio):
    #  设置随机数种子，保证每次生成的结果都是一样的
    np.random.seed(42)

    #  permutation随机生成0-len(data)随机序列
    shuffled_indices = np.random.permutation(len(data))

    #  按照比例ratio生成索引值列表
    head_set_size = int(len(data) * ratio)
    head_indices = shuffled_indices[:head_set_size]
    rear_indices = shuffled_indices[head_set_size:]

    #  按照比例随机生成集合
    head_set = []
    for i in range(len(head_indices)):
        head_set.append(data[head_indices[i]])
    rear_set = []
    for i in range(len(rear_indices)):
        rear_set.append(data[rear_indices[i]])

    return head_set, rear_set


def split_train_set(set_path):
    print('Spliting ' + set_path)

    #  获取所有采样帧图片
    files = []
    for f in os.listdir(set_path):
        if not os.path.isdir(f):
            files.append(f)

    #  将数据集按比例分配 6 2 2 
    train_set, test_set = split_data(files, 0.8)
    train_set, validation_set = split_data(train_set, 0.75)

    #  移动文件
    for train in train_set:
        move_file(set_path + '/' + train, 'train/' + set_path)
    for validation in validation_set:
        move_file(set_path + '/' + validation, 'validation/' + set_path)
    for test in test_set:
        move_file(set_path + '/' + test, 'test/' + set_path)
    

def run():
    gest_dirs = []
    for d in os.listdir('.'):
        if os.path.isdir(d) and d != 'CV_Test':
            gest_dirs.append(d)
    
    for d in gest_dirs:
        split_train_set(d)


if __name__ == "__main__":
    run()
