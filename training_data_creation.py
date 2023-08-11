# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import glob, math
import pickle 
import random

levels = []#list of dictionaries, each dictionary a level

characters = ["Q", "B", "C", "<", "-", "R",  "W", "S"]

#Load SMK Levels
for levelFile in glob.glob("./Processed/*.txt"):
  print ("Processing: "+levelFile)
  with open(levelFile) as fp:
    level = {}
    y = 0
    for line in fp:
      level[y] = {}
      for x in range(0, len(line)-1):
        onehot = [0]*len(characters)
        onehot[characters.index(line[x])] =1
        level[y][x]= onehot
      y+=1
    levels.append(level)


X = []
Y = []
for level in levels:
  for x in range(0, len(level[0].keys())-13):
    chunk = []
    for y in range(0, 14):
      line = []
      for xi in range(0, 14):
        line.append(level[y][x+xi])

      chunk.append(line)
    X.append(chunk)
    Y.append(chunk)


newXTrainingData = []
newYTrainingData = []
'''
#Reprocess them
for x in X: 
  currX = list(x)
  
  for i in range(0, 10):
    cloneX = list(currX)
    additionsY = []

    for x in range(0, 14):
      line = []
      for y in range(0, 14):
        line.append([0]*len(characters))
      additionsY.append(line)
      
    for j in range (0, 10):
      randXCoordinate = random.randrange(0,14)
      randYCoordinate = random.randrange(0, 14)

      while sum(cloneX[randXCoordinate][randYCoordinate])==0:
        randXCoordinate = random.randrange(0,14)
        randYCoordinate = random.randrange(0, 14)

      additionsY[randXCoordinate][randYCoordinate] = cloneX[randXCoordinate][randYCoordinate]
      cloneX[randXCoordinate][randYCoordinate] = [0]*len(characters)

    newXTrainingData.append(cloneX)
    newYTrainingData.append(additionsY)

    currX= cloneX

'''
def onehot_encode(value, categories):
    onehot_array = [0] * len(categories)
    index = categories.index(value)
    onehot_array[index] = 1
    return onehot_array

#Tilebased Reprocessing

for x in X:
    currX = list(x)
    desiredPosition = 0
    while desiredPosition < 8: 
        #cloneX = list(x)
        everRemoveAnything = False

        cloneX = []
        for y in range(0, 14):
            line = []
            for xi in range(0, 14):
                line.append(currX[y][xi])
            cloneX.append(line)
        
        
        additionsY = []
        for y in range(0, 14):
            line = []
            for xi in range(0, 14):
                line.append([0]*8)
            additionsY.append(line)
        
        for y in range (0, 14):
            for xi in range(0, 14):
            
                #if np.array_equal(cloneX[xi][y], onehot_encode(characters[desiredPosition], characters)):
                if 1 in cloneX[y][xi] and cloneX[y][xi].index(1) == desiredPosition:
                    everRemoveAnything = True
                    additionsY[y][xi] = list(cloneX[y][xi])
                    cloneX[y][xi] = [0]*len(characters)
        if everRemoveAnything:
            newXTrainingData.append(cloneX)
            newYTrainingData.append(list(additionsY))
            currX = cloneX
            

        desiredPosition +=1


X = np.array(newXTrainingData)
#print(len(newYTrainingData))
Y = np.array(newYTrainingData)

#Visualize State
for j in range(0, len(newXTrainingData)):
    chunkTrue = ""
    for y in range(0, 14):
        line = ""
        for x in range(0, 14):
            maxIndex = -1
            maxValue = 0
            if sum(newXTrainingData[j][y][x])==0:
                line += " "
            else:
                for i in range(0, len(newXTrainingData[j][y][x])):
                    if newXTrainingData[j][y][x][i]>maxValue:
                        maxValue = newXTrainingData[j][y][x][i]
                        maxIndex = i
                line += characters[maxIndex]
        line +="\n"
        chunkTrue+=line

    print("True State")
    print(chunkTrue)

    chunkTrue = ""
    for y in range(0, 14):
        line = ""
        for x in range(0, 14):
            maxIndex = -1
            maxValue = 0
            if sum(newYTrainingData[j][y][x])==0:
                line += " "
            else:
                for i in range(0, len(newYTrainingData[j][y][x])):
                    if newYTrainingData[j][y][x][i]>maxValue:
                        maxValue = newYTrainingData[j][y][x][i]
                        maxIndex = i
                line += characters[maxIndex]
        line +="\n"
        chunkTrue+=line

    print("True Action")
    print(chunkTrue)

X = np.reshape(X, (-1, 14, 14, 8))
print(X.shape)

pickle.dump(X, open("smk_trainRandX.p", "wb"))
pickle.dump(Y, open("smk_trainRandY.p", "wb"))
