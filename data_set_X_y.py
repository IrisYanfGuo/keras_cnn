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
    a = pickle.load(open(path[0],'rb')).values

    len1 = a.shape[0]
    len2 = a.shape[1]
    X = np.ndarray(shape=(len(path),1,len1,len2))
    for i in range(len(path)):
        X[i,0] = pickle.load(open(path[i],'rb')).values
   #X = np.ndarray(X)
    return X,(label,path)







def main():
    t = time.time()
    STFTdir = read_pickle_dir("./file_list.txt", "./STFTPickle")
    sel_word = STFTdir['word'].unique()
    STFTdir = STFTdir[STFTdir['word'].isin(sel_word)]
    STFTdir.index = np.arange(0, len(STFTdir))
    X, label = load_data(STFTdir)
    t2 = time.time()
    print("loading time:", (t2 - t))

    # divide training set and test set
    X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=42)
    print(X_train.shape)

    xtrainf = open("./X_train_all.pkl_0.7", 'wb')
    ytrainf = open("./y_train_all.pkl_0.7", 'wb')
    xtestf = open("./X_test.pkl_all_0.7", 'wb')
    ytestf = open("./y_test.pkl_all_0.7", 'wb')
    pickle.dump(X_train, xtrainf)
    pickle.dump(y_train, ytrainf)
    pickle.dump(X_test, xtestf)
    pickle.dump(y_test, ytestf)
    xtrainf.close()
    ytrainf.close()
    xtestf.close()
    ytestf.close()





    # print(X[0])


if __name__ == '__main__':
    main()