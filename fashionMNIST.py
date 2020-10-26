import tensorflow as tf 
from tensorflow import keras

import numpy as np 

fashion_mnist = keras.datasets.fashion_mnist
housing = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print(housing.load_data())