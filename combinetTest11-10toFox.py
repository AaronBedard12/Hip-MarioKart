# -*- coding: utf-8 -*-




""" Cifarnet.
https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf

['conv2d_1W', (5, 5, 3, 64)]
['fc_2b', (192,)]
['fc_3b', (11,)]
['fc_3W', (192, 11)]
['fc_1W', (4096, 384)]
['conv2d_2b', (64,)]
['fc_2W', (384, 192)]
['conv2d_1b', (64,)]
['fc_1b', (384,)]
['conv2d_2W', (5, 5, 64, 64)]

"""

from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import numpy as np
import random
from scipy import misc
import pickle, glob

def unpickle(file):
    import cPickle
    with open(file, 'rb') as fo:
        dict = cPickle.load(fo)
    return dict


#Params
numMaxFoxes = 250


#TRAINING DATA
dictionaries = []
options = []
for i in range(1,6):
  d = unpickle("./data_batch_"+str(i))
  dictionaries.append(d)
  X = list(d["data"])
  Y = []
  for val in d["labels"]:
    onehot = [0,0,0,0,0,0,0,0,0,0,0]
    onehot[val] =1
    Y.append(onehot)
  options.append([X,Y])

#TRAIN FOX DATA
numLoadedFoxes = 0
d = unpickle("./cifar-100-python/train")
for i in range(0,50000):
  if d["fine_labels"][i] == 60:
    if numLoadedFoxes<numMaxFoxes:
      numLoadedFoxes+=1.0
      xData =d["data"][i]
      yData = [0,0,0,0,0,0,0,0,0,0,1]
      for option in options:
        option[0].append(xData)
        option[1].append(yData)

#TESTING DATA
testDict = unpickle("./test_batch")
testX = list(testDict["data"])
testY = []
for val in testDict["labels"]:
  onehot = [0,0,0,0,0,0,0,0,0,0,0]
  onehot[val] =1
  testY.append(onehot)

#TEST FOX DATA
d = unpickle("./cifar-100-python/test")
for i in range(0,10000):
  if d["fine_labels"][i] == 60:
    testX.append(d["data"][i])
    onehot = [0,0,0,0,0,0,0,0,0,0,1]
    testY.append(onehot)

#Reshape stuff
for option in options:
  option[0] = np.array(option[0])
  option[0] = np.reshape(option[0], [-1, 32, 32, 3])
  option[1] = np.array(option[1])

testX = np.array(testX)
testX = np.reshape(testX, [-1, 32, 32, 3])
testY = np.array(testY)

# Building 'Cifarnet'
network = input_data(shape=[None, 32, 32, 3])
conv1 = conv_2d(network, 64, 5, activation='relu')
network = max_pool_2d(conv1, 2, strides=2)
network = local_response_normalization(network)
conv2 = conv_2d(network, 64, 5, activation='relu')
network = max_pool_2d(conv2, 2, strides=2)
network = local_response_normalization(network)
fc1 = fully_connected(network, 384, activation='tanh')
network = dropout(fc1, 0.5)
fc2 = fully_connected(network, 192, activation='tanh')
network = dropout(fc2, 0.5)
fc3 = fully_connected(network, 11, activation='softmax')
network = regression(fc3, optimizer='momentum',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Training
model = tflearn.DNN(network, checkpoint_path='model_cifar10b',
                    max_checkpoints=1, tensorboard_verbose=2)

fileNameToVariable = {}
fileNameToVariable["conv2d_1W"] = conv1.W
fileNameToVariable["conv2d_1b"] = conv1.b
fileNameToVariable["conv2d_2W"] = conv2.W
fileNameToVariable["conv2d_2b"] = conv2.b
fileNameToVariable["fc_1W"] = fc1.W
fileNameToVariable["fc_1b"] = fc1.b
fileNameToVariable["fc_2W"] = fc2.W
fileNameToVariable["fc_2b"] = fc2.b
fileNameToVariable["fc_3W"] = fc3.W
fileNameToVariable["fc_3b"] = fc3.b

fileNameToLoadedValue = {}
fileNameToLoadedValue["conv2d_1W"] = []
fileNameToLoadedValue["conv2d_1b"] = []
fileNameToLoadedValue["conv2d_2W"] = []
fileNameToLoadedValue["conv2d_2b"] = []
fileNameToLoadedValue["fc_1W"] = []
fileNameToLoadedValue["fc_1b"] = []
fileNameToLoadedValue["fc_2W"] = []
fileNameToLoadedValue["fc_2b"] = []
fileNameToLoadedValue["fc_3W"] = []
fileNameToLoadedValue["fc_3b"] = []

modelFolder = "./model2-weights/"

#Get weights
#TODO; load in weights of a trained tflearn model on cifar10
#model.load(NAME OF TRAINED MODEL)

#TESTING
for key in fileNameToLoadedValue.keys():
  weights = pickle.load(open(modelFolder+key+".p", "rb"))
  fileNameToLoadedValue[key] = weights

#Determine expansion instructions per each variable
expansionInstructions = {}
expansionInstructions["conv2d_1W"] = []
expansionInstructions["conv2d_1b"] = []
expansionInstructions["conv2d_2W"] = []
expansionInstructions["conv2d_2b"] = []
expansionInstructions["fc_1W"] = []
expansionInstructions["fc_1b"] = []
expansionInstructions["fc_2W"] = []
expansionInstructions["fc_2b"] = []
expansionInstructions["fc_3W"] = []
expansionInstructions["fc_3b"] = []

#Network and Expansion shapes
networkShapes = {}
networkShapes['conv2d_1W'] = (5, 5, 3, 64)
networkShapes['fc_2b']=(192)
networkShapes['fc_3b'] = (11)
networkShapes['fc_3W'] = (192, 11)
networkShapes['fc_1W'] = (4096, 384)
networkShapes['conv2d_2b'] = (64)
networkShapes['fc_2W'] = (384, 192)
networkShapes['conv2d_1b'] = (64)
networkShapes['fc_1b'] =  (384)
networkShapes['conv2d_2W'] = (5, 5, 64, 64)

expansionShapes = {}
expansionShapes['conv2d_1W'] = (64, 3, 5, 5)
expansionShapes['fc_2b']=(192)
expansionShapes['fc_3b'] = (11)
expansionShapes['fc_3W'] = (11, 192)
expansionShapes['fc_1W'] = (384, 4096)
expansionShapes['conv2d_2b'] = (64)
expansionShapes['fc_2W'] = (192, 384)
expansionShapes['fc_1b'] =  (384)
expansionShapes['conv2d_1b'] = (64)
expansionShapes['conv2d_2W'] = (64, 64, 5, 5)


maxBest = [0.75,0,0]
maxExpansion = {}
for key in expansionInstructions.keys():
  maxExpansion[key] = expansionInstructions[key]


for metaAttempts in range(0, 10):
  #ALL ONES STARTING POINT
  expansionInstructions['conv2d_1W'] = []
  expansionInstructions['fc_2b']= []
  expansionInstructions['fc_3b'] = []
  expansionInstructions['fc_3W'] = []
  expansionInstructions['fc_1W'] = []
  expansionInstructions['conv2d_2b'] = []
  expansionInstructions['fc_2W'] = []
  expansionInstructions['conv2d_1b'] = []
  expansionInstructions['conv2d_2W'] =[]
  expansionInstructions['fc_1b'] =  []

  #Build up initial instructions
  for key in expansionInstructions.keys():
    onesArray = None
    randArray = np.random.random_sample(expansionShapes[key])
    onesArray = np.ones(expansionShapes[key])
    mixArray = randArray
    for i in range(0, 5):
      mixArray+=onesArray
    mixArray = np.true_divide(mixArray,5.0)
    if not isinstance(expansionShapes[key], int):
      for i in range(0,expansionShapes[key][0]):
        if key == "fc_3W":
          if i==10:
            #A MIX OF THREE AND FIVE
            listToUse = []
           
            #listToUse.append([i, randArray[i]])
            #With 500
            if numLoadedFoxes==500:
              listToUse.append([0,np.true_divide(mixArray[0], 1.0)])
              listToUse.append([8,np.true_divide(mixArray[8], 2.0)])
              listToUse.append([2,np.true_divide(mixArray[2], 2.0)])
              listToUse.append([4,np.true_divide(mixArray[4], 2.0)])
            elif numLoadedFoxes==250:
              listToUse.append([0,np.true_divide(mixArray[0], 1.0)])
              listToUse.append([4,np.true_divide(mixArray[4], 2.0)])
              listToUse.append([2,np.true_divide(mixArray[2], 2.0)])
              listToUse.append([8,np.true_divide(mixArray[8], 2.0)])
              listToUse.append([9,np.true_divide(mixArray[9], 8.0)])
              listToUse.append([7,np.true_divide(mixArray[7], 4.0)])
              listToUse.append([5,np.true_divide(mixArray[5], 16.0)])
            elif numLoadedFoxes==100:
              listToUse.append([0,np.true_divide(mixArray[0], 1.0)])
              listToUse.append([4,np.true_divide(mixArray[4], 2.0)])
              listToUse.append([2,np.true_divide(mixArray[2], 4.0)])
              listToUse.append([8,np.true_divide(mixArray[8], 2.0)])
              listToUse.append([3,np.true_divide(mixArray[3], 8.0)])
              listToUse.append([9,np.true_divide(mixArray[9], 8.0)])
              listToUse.append([7,np.true_divide(mixArray[7], 8.0)])
            elif numLoadedFoxes==50:
              listToUse.append([0,np.true_divide(mixArray[0], 1.0)])
              listToUse.append([4,np.true_divide(mixArray[4], 2.0)])
              listToUse.append([7,np.true_divide(mixArray[7], 4.0)])
              listToUse.append([8,np.true_divide(mixArray[8], 2.0)])
            elif numLoadedFoxes==10:
              listToUse.append([0,np.true_divide(mixArray[0], 1.0)])
              listToUse.append([4,np.true_divide(mixArray[4], 2.0)])
              listToUse.append([8,np.true_divide(mixArray[8], 2.0)])
              listToUse.append([7,np.true_divide(mixArray[7], 4.0)])
            elif numLoadedFoxes==5:
              listToUse.append([0,np.true_divide(mixArray[0], 1.0)])
              listToUse.append([7,np.true_divide(mixArray[7], 3.0)])
              listToUse.append([8,np.true_divide(mixArray[8], 3.0)])
            else:
              listToUse.append([5,np.true_divide(mixArray[5], 1.0)])
            expansionInstructions[key].append(listToUse)
          else:
            expansionInstructions[key].append([[i,mixArray[i]]])
        else:
          expansionInstructions[key].append([[i,onesArray[i]]])
    else:
      if key == "fc_3b":
        expansionInstructions[key].append([[0,mixArray]])
      else:
        expansionInstructions[key].append([[0,onesArray]])

  conceptualExpansion = {}
  for key in expansionInstructions.keys():
    conceptualExpansion[key] = expansionInstructions[key]

  #Construct the weights for this combinet from the expansion
  for var in fileNameToVariable.keys():
    finalWeights = []
    trueWeights = fileNameToLoadedValue[var]
    loadedWeights = trueWeights.transpose()
    if not isinstance(expansionShapes[var], int):
      for indexPairs in conceptualExpansion[var]:
        thisWeights = None
        thisWeightsLoaded = False
        for indexPair in indexPairs:

          if not thisWeightsLoaded:
            thisWeights = np.copy(loadedWeights[indexPair[0]])*indexPair[1]
            thisWeightsLoaded = True
          else:
            weights = np.copy(loadedWeights[indexPair[0]])*indexPair[1]
            thisWeights +=weights
        finalWeights.append(thisWeights)
    else:        
      for indexPair in conceptualExpansion[var]:
        weightCheck = np.copy(loadedWeights)
        if var=="fc_3b":
          weightCheck = list(weightCheck)
          weightCheck.append(weightCheck[5])
          weightCheck = np.array(weightCheck)
        finalWeights = weightCheck*indexPair[0][1]

    finalWeights = np.array(finalWeights)
    finalWeights = finalWeights.transpose()
    model.set_weights(fileNameToVariable[var], finalWeights)

  
  #TESTING INITS
  option = random.choice(options)
  Xtouse = option[0]
  Ytouse = option[1]
  predY = model.predict(Xtouse)

  numCorrect = 0
  numFoxCorrect = 0
  foxIndexes = []
  foxValues = []

  nonFoxIndexes = []
  nonFoxValues = []
  for i in range(0, len(predY)):
    maxVal = max(predY[i])
    maxIndex = list(predY[i]).index(maxVal)

    trueIndex = list(Ytouse[i]).index(1)
    if maxIndex==trueIndex:
      if trueIndex==10:
        numFoxCorrect+=1
      else:
        numCorrect+=1
    else:
      if maxIndex==10:
        if trueIndex in nonFoxIndexes:
          nonFoxValues[nonFoxIndexes.index(trueIndex)] +=1
        else:
          nonFoxIndexes.append(trueIndex)
          nonFoxValues.append(1)

    if trueIndex==10:
      
      if maxIndex in foxIndexes:
        foxValues[foxIndexes.index(maxIndex)] +=1
      else:
        foxIndexes.append(maxIndex)
        foxValues.append(1)
    
  print ([numCorrect, numFoxCorrect])
  print (foxIndexes)
  print (foxValues)
  print (nonFoxIndexes)
  print (nonFoxValues)
    
  #Start greedy search
  bestScores = [(numCorrect+numFoxCorrect)/(10000.0+numLoadedFoxes),0,0]

  if(numFoxCorrect==0):
    bestScores=[0.5,0,0]

  #500 random attempts of hill climbing
  greater = 0
  noFoxes = 0
  for attempts in range(0, 100):
    print ("     Attempts: "+str(attempts))
    #Create a random child
    childExpansion = {}

    #copy over the existing conceptualExpansion
    for var in conceptualExpansion.keys():
      childExpansion[var] = []

      for indexPairs in conceptualExpansion[var]:
        listAtIndex = []
        for indexPair in indexPairs:
          listAtIndex.append([indexPair[0], np.copy(indexPair[1])])
        childExpansion[var].append(listAtIndex)

    
    #Make N random alterations 
    numChanges = 1

    for n in range(0,numChanges):
      var = random.choice(conceptualExpansion.keys())
      choice = np.random.random_sample()
      if choice<0.33:
        #Change a random filter
        indexToChange = [np.random.randint(len(childExpansion[var]))]
        indexToChange.append(np.random.randint(len(childExpansion[var][indexToChange[0]])))
        for k in range(0, len(childExpansion[var][0][0][1].shape)):
          indexToChange.append(np.random.randint(childExpansion[var][0][0][1].shape[k]))
          
          if len(indexToChange)==3:
            conceptualExpansion[var][indexToChange[0]][0][1][indexToChange[2]]= np.random.random_sample()
          elif len(indexToChange)==4:
            conceptualExpansion[var][indexToChange[0]][0][1][indexToChange[2]][indexToChange[3]]= np.random.random_sample()
          elif len(indexToChange)==5:
            conceptualExpansion[var][indexToChange[0]][0][1][indexToChange[2]][indexToChange[3]][indexToChange[4]]= np.random.random_sample()
          elif len(indexToChange)==6:
            conceptualExpansion[var][indexToChange[0]][0][1][indexToChange[2]][indexToChange[3]][indexToChange[4]][indexToChange[5]]= np.random.random_sample()
      elif choice<0.67:
        if not isinstance(expansionShapes[var], int):
          #Add an index
          onesArray = np.ones(expansionShapes[var])
          randomIndex = np.random.randint(len(childExpansion[var]))#Index to add to
          randomIndex2 = np.random.randint(expansionShapes[var][0])#Index to add
          if var=="fc_3W":
            if randomIndex2==10:
              randomIndex2-=1
          childExpansion[var][randomIndex].append([randomIndex2,onesArray[randomIndex2]])

      else:
        if not isinstance(expansionShapes[var], int):
            #swap an index
            randomIndex = np.random.randint(len(childExpansion[var]))#Index to swap 
            randomIndex2 = np.random.randint(len(childExpansion[var][randomIndex]))#Index  at that index to swap
            randomIndex3 = np.random.randint((expansionShapes[var][0]))#Index to swap to

            if var=="fc_3W":
              if randomIndex3==10:
                randomIndex3-=1
            childExpansion[var][randomIndex][randomIndex2][0] = randomIndex3

    #model.load(modelToBase)
    #Construct the weights for this combinet from the expansion
    for var in fileNameToVariable.keys():
      finalWeights = []
      trueWeights = fileNameToLoadedValue[var]
      loadedWeights = trueWeights.transpose()#np.reshape(, expansionShapes[var])
      if not isinstance(expansionShapes[var], int):
        for indexPairs in childExpansion[var]:
          thisWeights = None
          thisWeightsLoaded = False
          for indexPair in indexPairs:

            if not thisWeightsLoaded:
              if indexPair[0]<loadedWeights.shape[0]:
                thisWeights = np.copy(loadedWeights[indexPair[0]])*indexPair[1]
                thisWeightsLoaded = True
            else:
              weights = np.copy(loadedWeights[indexPair[0]])*indexPair[1]
              thisWeights +=weights
          finalWeights.append(thisWeights)
      else:  
        weightCheck = np.copy(loadedWeights) 
        if var=="fc_3b":
          weightCheck = list(weightCheck)
          weightCheck.append(weightCheck[5])
          weightCheck = np.array(weightCheck)
        for indexPair in childExpansion[var]:
          finalWeights = weightCheck*indexPair[0][1]

      finalWeights = np.array(finalWeights)
      finalWeights = finalWeights.transpose()#np.reshape(finalWeights, networkShapes[var])
      model.set_weights(fileNameToVariable[var], finalWeights)

    #Test this combinet
    print ("     Predict on new model")

    option = random.choice(options)
    Xtouse = option[0]
    Ytouse = option[1]
    predY = model.predict(Xtouse)

    numCorrect = 0
    numFoxCorrect = 0
    possibleFoxCorrects = 0
    for i in range(0, len(predY)):
      maxVal = max(predY[i])
      maxIndex = list(predY[i]).index(maxVal)

      trueIndex = list(Ytouse[i]).index(1)

      if trueIndex==10:
        possibleFoxCorrects+=1

      if maxIndex==trueIndex:
        if trueIndex==10:
          numFoxCorrect+=1
        else:
          numCorrect+=1

    if numFoxCorrect>0:
      noFoxes = 0
    #print ("Possible Fox Corrects: "+str(possibleFoxCorrects))
    performance = (numCorrect+numFoxCorrect)/(10000.0+numLoadedFoxes)
    print ("     Score: "+str([performance,numCorrect,numFoxCorrect]))
    if performance>bestScores[0] and numFoxCorrect>0:#possibleFoxCorrects*0.5:
      greater = 0
      print ("     Child was better")
      bestScores = [performance,numCorrect,numFoxCorrect]
      for key in conceptualExpansion.keys():
        conceptualExpansion[key] = childExpansion[key]

    else:
      greater+=1
      if attempts==0:#break if no better than baseline bestScores
        break
      elif numFoxCorrect==0 and attempts==1:#break if no foxes correct
        break
      elif numFoxCorrect==0:
        noFoxes+=1
        if noFoxes>5:
          print ("No foxes for 5 attempts")
          break

      if greater>10:
        print ("Stopped improving for 10 neighbors")
        break
    
  
  print ("Training results: "+str(bestScores))
  if bestScores[0]>maxBest[0]:
    print ("New Max")
    maxBest = bestScores
    for key in conceptualExpansion.keys():
      maxExpansion[key] = conceptualExpansion[key]
  
  #except:
  #  print("Brokened")
  

#SAVE OUT LAST BIT
#model.load(modelToBase)

#CONSTRUCT COMBINET FROM MAX COMBINATION
#model.load(modelToBase)
#Construct the weights for this combinet from the expansion
for var in fileNameToVariable.keys():
  finalWeights = []
  trueWeights = fileNameToLoadedValue[var]
  loadedWeights = trueWeights.transpose()#np.reshape(, expansionShapes[var])
  if not isinstance(expansionShapes[var], int):
    for indexPairs in childExpansion[var]:
      thisWeights = None
      thisWeightsLoaded = False
      for indexPair in indexPairs:

        if not thisWeightsLoaded:
          if indexPair[0]<loadedWeights.shape[0]:
            thisWeights = np.copy(loadedWeights[indexPair[0]])*indexPair[1]
            thisWeightsLoaded = True
        else:
          weights = np.copy(loadedWeights[indexPair[0]])*indexPair[1]
          thisWeights +=weights
      finalWeights.append(thisWeights)
  else:  
    weightCheck = np.copy(loadedWeights) 
    if var=="fc_3b":
      weightCheck = list(weightCheck)
      weightCheck.append(weightCheck[5])
      weightCheck = np.array(weightCheck)
    for indexPair in childExpansion[var]:
      finalWeights = weightCheck*indexPair[0][1]

  finalWeights = np.array(finalWeights)
  finalWeights = finalWeights.transpose()#np.reshape(finalWeights, networkShapes[var])
  model.set_weights(fileNameToVariable[var], finalWeights)


#TESTING INITS
predY = model.predict(testX)

numCorrect = 0
numFoxCorrect = 0
foxIndexes = []
foxValues = []

nonFoxIndexes = []
nonFoxValues = []
for i in range(0, len(predY)):
  maxVal = max(predY[i])
  maxIndex = list(predY[i]).index(maxVal)

  trueIndex = list(testY[i]).index(1)
  if maxIndex==trueIndex:
    if trueIndex==10:
      numFoxCorrect+=1
    else:
      numCorrect+=1
  else:
    if maxIndex==10:
      if trueIndex in nonFoxIndexes:
        nonFoxValues[nonFoxIndexes.index(trueIndex)] +=1
      else:
        nonFoxIndexes.append(trueIndex)
        nonFoxValues.append(1)

  if trueIndex==10:
    
    if maxIndex in foxIndexes:
      foxValues[foxIndexes.index(maxIndex)] +=1
    else:
      foxIndexes.append(maxIndex)
      foxValues.append(1)
  
print ("Test values: "+str([numCorrect, numFoxCorrect]))
print ("Fox indexes")
print (foxIndexes)
print (foxValues)
print("Non fox")
print (nonFoxIndexes)
print (nonFoxValues)
model.save('combinet11-10ToFox'+str(numLoadedFoxes)+'.tflearn')
