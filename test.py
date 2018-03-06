#! /usr/bin/env python3
'''
Created by Yanfang Guo on 22/02/2018, 6:03 PM
@yanfguo@outlook.com
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import pickle
from keras.preprocessing.image import ImageDataGenerator, DirectoryIterator
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder


# ======= load dataset==========
f = open("./pickleFile/data10Mar06.pkl", 'rb')
dataset = pickle.load(f)

STFT = dataset['STFT'].as_matrix()
X = np.ndarray((len(STFT), STFT[0].shape[0], STFT[0].shape[1], STFT[0].shape[2]))

for i in range(len(STFT)):
    X[i] = STFT[i]

y = dataset['label'].values
path = dataset['path'].values

X_train, X_test, y_train, y_test, path_train, path_test = train_test_split(X, y, path, test_size=0.3, random_state=42)

# ====== load dataset completed=========

# encoder
encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
y_train = np_utils.to_categorical(encoded_Y)
encoder2 = LabelEncoder()
encoder2.fit(y_test)
encoded2_Y = encoder.transform(y_test)
y_test = np_utils.to_categorical(encoded2_Y)




def main():
    f = open("./pickleFile/data30Mar06.pkl", 'rb')
    dataset = pickle.load(f)
    print(type(dataset))


if __name__ == '__main__':
    main()
