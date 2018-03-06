#! /usr/bin/env python3
'''
Created by Yanfang Guo on 06/03/2018, 5:02 PM
@yanfguo@outlook.com
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydot
import tensorflow as tf
import pickle
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from keras.layers import MaxPooling2D, ZeroPadding2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras import backend as K
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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

encoded2_Y = encoder.transform(y_test)
y_test = np_utils.to_categorical(encoded2_Y)

print(encoded2_Y)
print(len(encoded2_Y))

from keras.utils import plot_model
from keras.models import load_model

model = load_model("./cat10.h5")

out = model.predict(X_test[:1000])
labelMax = np.argmax(out, axis=1)
out_y = encoder.inverse_transform(labelMax)
print(out_y)
real_y = encoder.inverse_transform(encoded2_Y[:1000])

f = open("./0catfalse.txt", 'w')
n = 0
for i in range(len(out_y)):
    if out_y[i] != real_y[i]:
        n+=1
        str1 = ""
        str1 += "predict label: " + out_y[i] + " real label: " + real_y[i] + "\npath:" + path_test[i] + "\n"
        str1 += "predict proportition:" + str(out[i])
        str1 += "\n\n"
        f.write(str1)
f.close()
print(n)


def main():
    print("hello world")


if __name__ == '__main__':
    main()
