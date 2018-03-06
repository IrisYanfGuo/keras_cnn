#! /usr/bin/env python3
'''
Created by Yanfang Guo on 28/02/2018, 3:14 PM
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




from keras.utils import plot_model
from keras.models import load_model
model = load_model("./cat10.h5")
plot_model(model,to_file="./model.png",show_layer_names=True,show_shapes=True)
a = model.get_weights()
model.summary()
#img_to_visualize = x_train[65]

def layer_to_visualize(layer):
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])

    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)

    print('Shape of conv:', convolutions.shape)

    n = convolutions.shape[0]
    n = int(np.ceil(np.sqrt(n)))

    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(12, 8))
    for i in range(len(convolutions)):
        ax = fig.add_subplot(n, n, i + 1)
        ax.imshow(convolutions[i])
plt.figure()
plt.imshow(x_train[65][0])
img_to_visualize = np.expand_dims(img_to_visualize, axis=0)
layer_to_visualize(model.get_layer("max_pooling2d_1"))
plt.show()


def main():
    print("hello world")


if __name__ == '__main__':
    main()

