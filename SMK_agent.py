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

import glob, math
import numpy as np
import random as rand

characters = ["Q", "B", "C", "<", "-", "R",  "W", "S"]

trainX = pickle.load(open("smk_trainX.p", "rb"))  # 40 (width) x 15 (height) x 34 (SMB entities) (state)
trainY = pickle.load(open("smk_trainY.p", "rb"))  # 40 (width) x 15 (height) x 32 (SMB entities except the player and the flat) (value of the AI making a particular addition)



# Architecture
networkInput = tflearn.input_data(shape=[None, 14, 14, 8])
conv = conv_2d(networkInput, 8, 4, activation='leaky_relu')
conv2 = conv_2d(conv, 16, 3, activation='leaky_relu')
conv3 = conv_2d(conv2, 32, 3, activation='leaky_relu')
fc = tflearn.fully_connected(conv3, 14 * 14 * 8, activation='leaky_relu')
mapShape = tf.reshape(fc, [-1, 14, 14, 8])
network = tflearn.regression(mapShape, optimizer='adam', metric='R2', loss='mean_square', learning_rate=0.001)

model = tflearn.DNN(network)
namesToLoad = ["convW", "convb", "conv2W", "conv2b", "conv3W", "conv3b", "fcW", "fcb"]
layers = {}

for name in namesToLoad:
    layers[name] = pickle.load(open(name+".p", "rb"))
    #print (layers[name])

layersToLoad = [conv.W, conv.b, conv2.W, conv2.b, conv3.W, conv3.b, fc.W, fc.b]

marioCharacters =  ["Ground", "Stair", "Treetop", "Block", "Bar", "Koopa", "Koopa 2", "PipeBody", "Pipe", "Question", "Coin", "Goomba", "CannonBody", "Cannon", "Lakitu", "Bridge", "Hard Shell", "SmallCannon", "Plant", "Waves", "Hill", "Castle", "Snow Tree 2", "Cloud 2", "Cloud", "Bush", "Tree 2", "Bush 2", "Tree", "Snow Tree", "Fence", "Bark", "Flag", "Mario"]
correspondingValues = {
	"Q":"Question",
    "B":"Goomba",
    "C":"Coin",
    "<":"Pipe",
    "-":"Hill",
    "R":"Ground", 
	"W":"Block",
	"S":"Flag"
}

for i in range(0, len(layersToLoad)):
    newSMKWeightValues = np.zeros(layersToLoad[i].shape)
    layer = layersToLoad[i]
    #print(len(layer.shape))
    #print("\n" + str(namesToLoad[i]) + "\n")
    if "b" in namesToLoad[i]:
        for a in range(0, layer.shape[0]):
            newSMKWeightValues[a] = layers[namesToLoad[i]][a]
    elif "W" in namesToLoad[i] and "conv" in namesToLoad[i]:
        for a in range(0, layer.shape[0]):
            for b in range(0, layer.shape[1]):
                for c in range(0, layer.shape[2]):
                    for d in range(0, layer.shape[3]):
                        newSMKWeightValues[a,b,c,d] = layers[namesToLoad[i]][a,b,c,d]

                        # cValue = marioCharacters.index(correspondingValues[characters[c]])
                        # print(cValue)
                        # newSMKWeightValues[a,b,c,d] = layers[namesToLoad[i]][a,b,cValue,d]
    else:
        for a in range(0, layer.shape[0]):
            for b in range(0, layer.shape[1]):
                newSMKWeightValues[a,b] = layers[namesToLoad[i]][a,b]

    model.set_weights(layersToLoad[i], newSMKWeightValues)

model.fit(trainX,
          Y_targets=trainY,
          n_epoch=50,
          shuffle=True,
          show_metric=True,
          snapshot_epoch=False,
          batch_size= 20,
          run_id='cocreativeTest')

model.save("SMK_agent_transferV3.tflearn")
print("\n\n")
