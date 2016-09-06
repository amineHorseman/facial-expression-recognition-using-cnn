

import tensorflow as tf 
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression 

from parameters import NETWORK

def build_model():
    network = input_data(shape=[None, NETWORK.input_size, NETWORK.input_size, 1], name='input')
    network = conv_2d(network, 64, 5, activation=NETWORK.activation)
    #network = local_response_normalisatio(network)
    network = max_pool_2d(network, 3, strides = 2)
    network = conv_2d(network, 64, 5, activation=NETWORK.activation)
    network = max_pool_2d(network, 3, strides = 2)
    network = conv_2d(network, 128, 4, activation=NETWORK.activation)
    network = dropout(network, keep_prob=NETWORK.keep_prob)
    network = fully_connected(network, 3072, activation=NETWORK.activation)
    network = fully_connected(network, NETWORK.output_size, activation='softmax')
    network = regression(network, optimizer=NETWORK.optimizer, loss=NETWORK.loss, name='output')

    return network