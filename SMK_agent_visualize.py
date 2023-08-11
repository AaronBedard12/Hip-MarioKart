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
from PIL import Image

#Load sprites
sprites = {}
for filename in glob.glob("./Images/*.png"):
    print(filename)
    im = Image.open(filename)
    splits = filename.split("\\")
    name = splits[-1][:-4]
    sprites[name] = im


characters = ["Q", "B", "C", "<", "-", "R",  "W", "S"]

visualization = {}
visualization["W"] = "Wall_smk"
visualization["S"] = "StartingLine_smk"
visualization["-"] = "Grass_smk"
visualization["Q"] = "QuestionBlock_smk"
visualization["O"] = "Oil_smk"
visualization["C"] = "Coin_smk"
visualization["R"] = "Road_smk"
visualization["<"] = "Boost_smk"


np.random.seed(1)
random.seed(1)
tf.random.set_seed(1)

testX = pickle.load(open("smk_testTBX.p", "rb"))#40 (width) x 15 (height) x 34 (SMB entitites) (state)
testY = pickle.load(open("smk_testTBY.p", "rb"))#40 (width) x 15 (height) x 32 (SMB entities except the player and the flat) (value of the AI making a particular addition)

testX = np.reshape(testX, (-1, 14, 14, len(characters)))
testY = np.reshape(testY, (-1, 14, 14, len(characters)))

#Architecture
networkInput = tflearn.input_data(shape=[None, 14, 14, 8])
conv = conv_2d(networkInput, 8, 4, activation='leaky_relu')
conv2 = conv_2d(conv, 16, 3, activation='leaky_relu')
conv3 = conv_2d(conv2, 32, 3, activation='leaky_relu')
fc = tflearn.fully_connected(conv3, 14 * 14 * 8, activation='leaky_relu')
mapShape = tf.reshape(fc, [-1, 14, 14, 8])
network = tflearn.regression(mapShape, optimizer='adam', metric='R2', loss='mean_square', learning_rate=0.0001)
model = tflearn.DNN(network)

# model.load("SMK_agentTBR_transferV3.tflearn")




image = Image.new("RGB", (64, 112), color=(0, 0, 0))
pixels = image.load()

#Visualize 
for j in range(200, 210):
    transfer = random.randint(0,1)
    version = random.randint(1,3)
    
    if transfer == 1:
        model.load("SMK_agent_transferV" + str(version) +".tflearn")
    else: 
        model.load("SMK_agentV" + str(version) +".tflearn")

    testYPrime = model.predict(testX)

    pred = testYPrime[j]
    for y in range(0, 14):
        for x in range(0, 8):
            maxIndex = -1
            maxValue = 0
            for i in range(0, len(pred[y][x])):
                if pred[y][x][i]>maxValue:
                    maxValue = pred[y][x][i]
                    maxIndex = i

            imageToUse = None
            if characters[maxIndex] in visualization.keys():
                imageToUse = sprites[visualization[characters[maxIndex]]]
            if not imageToUse == None:
                pixelsToUse = imageToUse.load()
                for x2 in range(0, 8):
                    for y2 in range(0, 8):
                        if pixelsToUse[x2,y2][3]>0:
                            pixels[x*8+x2,y*8+y2] = pixelsToUse[x2,y2][0:-1]
    image.save("pred"+str(transfer)+str(j)+ ".jpeg", "JPEG")