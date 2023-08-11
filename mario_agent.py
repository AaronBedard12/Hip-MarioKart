'''
Agent Training Co-Creative Machine Learning
'''

import pickle
import csv, random, glob, random

import numpy as np
import tflearn
import tensorflow as tf
from tflearn import conv_2d, conv_3d, max_pool_2d, local_response_normalization, batch_normalization, fully_connected, \
    regression, input_data, dropout, custom_layer, flatten, reshape, embedding, conv_2d_transpose

import copy
import sys

#trainX = pickle.load(open("trainX.p", "rb"))#40 (width) x 15 (height) x 34 (SMB entitites) (state)
#trainY = pickle.load(open("trainY.p", "rb"))#40 (width) x 15 (height) x 32 (SMB entities except the player and the flat) (value of the AI making a particular addition)

#Architecture
networkInput = tflearn.input_data(shape=[None, 40, 15, 34])
conv = conv_2d(networkInput, 8,4, activation='leaky_relu')
conv2 = conv_2d(conv, 16,3, activation='leaky_relu')
conv3 = conv_2d(conv2, 32,3, activation='leaky_relu')
fc = tflearn.fully_connected(conv3, 40*15*32, activation='leaky_relu')
mapShape = tf.reshape(fc, [-1,40,15,32])
network = tflearn.regression(mapShape, optimizer='adam', metric='R2', loss='mean_square',learning_rate=0.0001)##

model = tflearn.DNN(network)
model.load("agent2.tf", weights_only=True)

layersToSave = [conv.W, conv.b, conv2.W, conv2.b, conv3.W, conv3.b, fc.W, fc.b]
namesToSave = ["convW", "convb", "conv2W", "conv2b", "conv3W", "conv3b", "fcW", "fcb"]

for i in range(0, len(layersToSave)):
    layer = layersToSave[i].eval(model.trainer.session)
    #print("Layer "+str(namesToSave[i])+" "+str(layer))
    zeros = np.zeros(layer.shape)
    
    
    if "b" in namesToSave[i]:
        for j in range(0, layer.shape[0]):
            zeros[j] = layer[j]
    elif "conv" in namesToSave[i] and "W" in namesToSave[i]:
        for a in range(0, layer.shape[0]):
            for b in range(0, layer.shape[1]):
                for c in range(0, layer.shape[2]):
                    for d in range(0, layer.shape[3]):
                        zeros[a,b,c,d] = layer[a,b,c,d]
    else: 
        for a in range(0, layer.shape[0]):
            for b in range(0, layer.shape[1]):
                zeros[a,b] = layer[a,b]

    
    pickle.dump(zeros, open(namesToSave[i]+".p", "wb"))



