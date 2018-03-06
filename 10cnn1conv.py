#! /usr/bin/env python3
'''
Created by Yanfang Guo on 22/02/2018, 11:43 AM
@yanfguo@outlook.com
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import time

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

f = open("./pickleFile/data10Mar06.pkl",'rb')
dataset = pickle.load(f)
print(type(dataset))

STFT = dataset['STFT'].as_matrix()
X = np.ndarray((len(STFT),STFT[0].shape[0],STFT[0].shape[1],STFT[0].shape[2]))
t = time.time()
for i in range(len(STFT)):
    X[i] = STFT[i]
t2 = time.time()
print("time for loop",(t2-t))

y = dataset['label'].values
path = dataset['path'].values

X_train, X_test, y_train, y_test,path_train,path_test = train_test_split(X, y,path, test_size=0.3, random_state=42)

print(X.shape)
# kaggle


encoder = LabelEncoder()
encoder.fit(y_train)
encoded_Y = encoder.transform(y_train)
y_train = np_utils.to_categorical(encoded_Y)

encoded2_Y = encoder.transform(y_test)
y_test = np_utils.to_categorical(encoded2_Y)

num_classes = y_test.shape[1]

print(X_train.shape)

def baseline_model():
    # create model
    model = Sequential()
    model.add(Conv2D(16, (5, 5), input_shape=(1,76,68), activation='relu',data_format='channels_first'))
    model.add(MaxPooling2D(pool_size=(3, 3),data_format='channels_first'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))
model.save("./cat10_simple.h5")
model.summary()

def main():
    print("hello world")


if __name__ == '__main__':
    main()
