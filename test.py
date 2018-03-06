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


def main():
    f = open("./pickleFile/XMar06.pkl", 'rb')
    # dataset = pickle.load(f)
    # print(type(dataset))
    # print(dataset[['STFT']].as_matrix().shape)


if __name__ == '__main__':
    main()
