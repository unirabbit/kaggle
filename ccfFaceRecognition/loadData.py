import os
import numpy as np
import matplotlib.image as mpimg


def list_all_files(rootdir):
    _files = []
    list = os.listdir(rootdir)  # 列出文件夹下所有的目录与文件
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        if os.path.isdir(path):
            _files.extend(list_all_files(path))
        if os.path.isfile(path):
            _files.append(path)
    return _files


def load_data(path):
    file_list = list_all_files(path)
    X_train = np.empty((len(file_list), 112, 96, 3), dtype='uint8')
    dict = {'African': 0, 'Asian': 1, 'Caucasian': 2, 'Indian': '3'}
    Y_train_list = []
    # x_train = mpimg.imread(path）
    for i, file in enumerate(file_list):
        X_train[i, :, :, :] = mpimg.imread(file)
        Y_train_list.append(dict[file.split('\\')[-3]])
    Y_train = np.array(Y_train_list)
    return X_train, Y_train


X_train, Y_train = load_data('./training')
print(f"X_train' shape is {X_train.shape}")
print(f"Y_train' shape is {Y_train.shape}")
