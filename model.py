"""
@AmineHorseman
Sep, 1st, 2016
"""
import tensorflow as tf 
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d 
from tflearn.layers.merge_ops import merge_outputs, merge
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression 
from tflearn.optimizers import Momentum, Adam

from parameters import NETWORK, HYPERPARAMS

def build_model(optimizer=HYPERPARAMS.optimizer, optimizer_param=HYPERPARAMS.optimizer_param, 
    learning_rate=HYPERPARAMS.learning_rate, keep_prob=HYPERPARAMS.keep_prob,
    learning_rate_decay=HYPERPARAMS.learning_rate_decay, decay_step=HYPERPARAMS.decay_step):

    images_network = input_data(shape=[None, NETWORK.input_size, NETWORK.input_size, 1], name='input1')
    images_network = conv_2d(images_network, 64, 5, activation=NETWORK.activation)
    #images_network = local_response_normalisatio(images_network)
    images_network = max_pool_2d(images_network, 3, strides = 2)
    images_network = conv_2d(images_network, 64, 5, activation=NETWORK.activation)
    images_network = max_pool_2d(images_network, 3, strides = 2)
    images_network = conv_2d(images_network, 128, 4, activation=NETWORK.activation)
    images_network = dropout(images_network, keep_prob=keep_prob)
    images_network = fully_connected(images_network, 1024, activation=NETWORK.activation)

    if NETWORK.use_landmarks:
        landmarks_network = input_data(shape=[None, 68, 2], name='input2')
        landmarks_network = fully_connected(landmarks_network, 1024, activation=NETWORK.activation)
        landmarks_network = fully_connected(landmarks_network, 40, activation=NETWORK.activation)
        images_network = fully_connected(images_network, 40, activation=NETWORK.activation)
        network = merge([images_network, landmarks_network], 'concat', axis=1)
    else:
        network = images_network
    network = fully_connected(network, NETWORK.output_size, activation='softmax')

    if optimizer == 'momentum':
        optimizer = Momentum(learning_rate=learning_rate, momentum=optimizer_param, 
                    lr_decay=learning_rate_decay, decay_step=decay_step)
    elif optimizer == 'adam':
        optimizer = Adam(learning_rate=learning_rate, beta1=optimizer_param, beta2=learning_rate_decay)
    else:
        print "Unknown optimizer: {}".format(optimizer)
    network = regression(network, optimizer=optimizer, loss=NETWORK.loss, learning_rate=learning_rate, name='output')

    return network