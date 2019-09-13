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
    file_list = list_all_files('./training/African')
    X_train = np.empty((len(file_list), 112, 96, 3), dtype='uint8')
    # x_train = mpimg.imread(path）
    for i, file in enumerate(file_list):
        X_train[i, :, :, :] = mpimg.imread(file)
    print(X_train.shape)

