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

characters = ["Q", "B", "C", "<", "-", "R",  "W", "S"]

testX = pickle.load(open("smk_testTBX.p", "rb"))  # 40 (width) x 15 (height) x 34 (SMB entities) (state)
testY = pickle.load(open("smk_testTBY.p", "rb"))  # 40 (width) x 15 (height) x 32 (SMB entities except the player and the flat) (value of the AI making a particular addition)

np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)

# Architecture
networkInput = tflearn.input_data(shape=[None, 14, 14, len(characters)])
conv = conv_2d(networkInput, 8, 4, activation='leaky_relu')
conv2 = conv_2d(conv, 16, 3, activation='leaky_relu')
conv3 = conv_2d(conv2, 32, 3, activation='leaky_relu')
fc = tflearn.fully_connected(conv3, 14 * 14 * len(characters), activation='leaky_relu')
mapShape = tf.reshape(fc, [-1, 14, 14, len(characters)])
network = tflearn.regression(mapShape, optimizer='adam', metric='R2', loss='mean_square', learning_rate=0.0001)

model = tflearn.DNN(network)
model.load("SMK_agent_transferV1.tflearn")
testYPrime = model.predict(testX)

correct = 0
for j in range(0, len(testYPrime)):
    pred = testYPrime[j]
    trueNext = testY[j]
    for y in range(0, 14):
        for x in range (0, 14):
            maxIndexPred = -1
            maxValuePred = 0
            maxIndexTruth = -1
            maxValueTruth = 0
            for i in range(0, len(pred[y][x])):
                if pred[y][x][i]>maxValuePred:
                    maxValuePred = pred[y][x][i]
                    maxIndexPred = i
                if trueNext[y][x][i]>maxValueTruth:
                    maxValueTruth = trueNext[y][x][i]
                    maxIndexTruth = i

            if maxIndexPred==maxIndexTruth:
                correct+=1
print("Accuracy: "+str(float(correct)/float(len(testYPrime))))
'''
#Visualize 
for j in range(0, len(testYPrime)):
    pred = testYPrime[j]
    chunk = ""
    for y in range(0, 14):
        line = ""
        for x in range(0, 14):
            maxIndex = -1
            maxValue = 0
            for i in range(0, len(pred[y][x])):
                if pred[y][x][i]>maxValue:
                    maxValue = pred[y][x][i]
                    maxIndex = i
            if maxIndex==-1:
                line+=" "
            else:
                line += characters[maxIndex]
        line +="\n"
        chunk+=line
    print("New Actions Predicted")
    print(chunk)

    trueNext = testY[j]

    chunkTrue = ""
    for y in range(0, 14):
        line = ""
        for x in range(0, 14):
            maxIndex = -1
            maxValue = 0
            if sum(testY[j][y][x])==0:
                line += " "
            else:
                for i in range(0, len(testY[j][y][x])):
                    if testY[j][y][x][i]>maxValue:
                        maxValue = testY[j][y][x][i]
                        maxIndex = i
                line += characters[maxIndex]
        line +="\n"
        chunkTrue+=line

    print("True State")
    print(chunkTrue)


    trueState = testX[j]

    chunkTrue = ""
    for y in range(0, 14):
        line = ""
        for x in range(0, 14):
            maxIndex = -1
            maxValue = 0
            if sum(testX[j][y][x])==0:
                line += " "
            else:
                for i in range(0, len(testX[j][y][x])):
                    if testX[j][y][x][i]>maxValue:
                        maxValue = testX[j][y][x][i]
                        maxIndex = i
                line += characters[maxIndex]
        line +="\n"
        chunkTrue+=line

    print("True Action")
    print(chunkTrue)
'''