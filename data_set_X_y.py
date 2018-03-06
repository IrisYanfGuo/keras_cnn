#! /usr/bin/env python3
'''
Created by Yanfang Guo on 22/02/2018, 5:26 PM
@yanfguo@outlook.com
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from read_dir_os import *


def load_data(dir):
    '''
    :param dir: the directory information of the data, type: pd.frame, columns=['word', 'speaker', 'path', 'temp']
    :return: type: X,label
    '''
    path = dir['path'].values
    label = dir['word'].values
    a = pickle.load(open(path[0], 'rb')).values

    len1 = a.shape[0]
    len2 = a.shape[1]
    #X = np.ndarray(shape=(len(path), 1, len1, len2))
    #for i in range(len(path)):
      #  X[i, 0] = pickle.load(open(path[i], 'rb')).values
        # X = np.ndarray(X)
    X = []
    for i in range(len(path)):
        STFT = np.ndarray((1, len1, len2))
        STFT[0] = pickle.load(open(path[i], 'rb')).values # shape = (1*len1*len2)
        path1 = path[i]
        label1 = label[i]
        X.append([STFT,label1,path1])

    X = pd.DataFrame(X,columns=['STFT', 'label', 'path'])

    return X


def main():
    t = time.time()
    STFTdir = read_pickle_dir("./file_list.txt", "./STFTPickle")
    sel_word = STFTdir['word'].unique()
    sel_word = sel_word[:2]
    STFTdir = STFTdir[STFTdir['word'].isin(sel_word)]
    STFTdir.index = np.arange(0, len(STFTdir))
    X = load_data(STFTdir)
    t2 = time.time()
    print("loading time:", (t2 - t))

    xfile = open("./pickleFile/X" + time.strftime("%b%d") + ".pkl", 'wb')
    pickle.dump(X, xfile)

    xfile.close()

    '''
    # divide training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=42)
    print(X_train.shape)

    '''





    # print(X[0])


if __name__ == '__main__':
    main()
    xfile = open("./pickleFile/X" + time.strftime("%b%d") + ".pkl", 'rb')
    xfile = pickle.load(xfile)
    print(xfile.head)
    print(xfile['STFT'].iloc[0].shape)


