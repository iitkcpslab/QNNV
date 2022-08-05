import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.datasets import mnist, fashion_mnist
from multiprocessing import Pool, cpu_count
import numpy as np


def normalizePixel(px):
    return px / 256


def flattenAndNormalizeImage(im):
    return normalizePixel(im.flatten())


def getFlattenedMNISTImages(fashion=False):
    if (fashion):
        (xTrain, yTrain), (xTest, yTest) = fashion_mnist.load_data()
    else:
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    x, y = np.concatenate((xTrain, xTest)), np.concatenate((yTrain, yTest))
    with Pool(processes=cpu_count()) as p:
        flatImages = p.map(flattenAndNormalizeImage, x, 100)
    return ((np.array(flatImages[:len(xTrain)]), np.array(y[:len(xTrain)])),
            (np.array(flatImages[len(xTrain):]), np.array(y[len(xTrain):])))
