# Chirag Rao Sahib      : 836011
# Maleakhi Agung Wijaya : 784091

###############################################################################

import numpy as np
import pandas as pd
from collections import defaultdict

SPLIT_RATIO = 0.8  # holdout ratio
ITERATIONS = 5  # iterations for unsupervised NB

DATASET1 = '2018S1-proj1_data/breast-cancer-dos.csv'
DATASET2 = '2018S1-proj1_data/car-dos.csv'
DATASET3 = '2018S1-proj1_data/hypothyroid-dos.csv'
DATASET4 = '2018S1-proj1_data/mushroom-dos.csv'
DATASETS = [DATASET1, DATASET2, DATASET3, DATASET4]
SAMPLE = '2018S1-proj1_data/sample.csv'  # example dataset

###############################################################################

def createKey(lst):
    '''
    helper function to create dict key given a list
    '''
    return '|'.join(str(i) for i in lst)

###############################################################################

def randDistGen(length):
    '''
    returns numpy array of 'length' random probabilities that sum to 1
    '''
    dist = np.random.random(length)
    return dist / dist.sum()

###############################################################################

def preprocessSup(data):
    '''
    read csv using pandas and partition data as per holdout
    '''
    dataFrame = pd.read_csv(data, header = None)
    split = np.random.rand(len(dataFrame)) < SPLIT_RATIO
    train = dataFrame[split]
    test = dataFrame[~split]

    return train, test

###############################################################################

def trainSup(trainSet):
    '''
    returns prior and posterior probabilities
    '''
    priorCounts = trainSet.iloc[:,-1].value_counts()
    priorProb = priorCounts / trainSet.shape[0]
    attribCount = trainSet.shape[1]
    posteriorProb = {}

    for attrib in range(attribCount - 1):  # class is not an attribute
        # generate list of unique attribute values and disregard ?
        attribValues = list(trainSet[attrib].unique())
        if ('?' in attribValues): attribValues.remove('?')

        # calculate posterior probabilities
        for c in priorCounts.index:
            for val in attribValues:
                # first filter by class then by attribute value
                filterClass = trainSet[trainSet.iloc[:,-1] == c]
                filterClassVal = filterClass[filterClass[attrib] == val]
                key = createKey([attrib, val, c])
                posteriorProb[key] = filterClassVal.shape[0] / priorCounts[c]

    return priorProb, posteriorProb

###############################################################################

def predictSup(testSet, priorProb, posteriorProb):
    '''
    returns predicted class given test data
    '''
    cleanTest = testSet.drop(testSet.columns[-1], axis=1)  # drop classes
    predictedClasses = []

    for i, instance in cleanTest.iterrows():
        currentMax = ['null', 0]  # track most probable class

        for c in priorProb.index:
            # maximum likelihood estimation of each instance
            prob = priorProb[c]

            for attrib, val in enumerate(list(instance)):
                key = createKey([attrib, val, c])
                if key in posteriorProb: prob *= posteriorProb[key]

            if prob > currentMax[1]: currentMax = [c, prob]

        # predicted class = most likely class
        predictedClasses.append(currentMax[0])

    return predictedClasses

###############################################################################

def evaluateSup(testSet, predictedClasses):
    '''
    simple accuracy of supervised NB
    '''
    correct = 0
    trueClass = testSet.iloc[:,-1].tolist()

    if len(trueClass) != len(predictedClasses):
        print('Error: Class length')
        return

    for i in range(len(trueClass)):
        if (trueClass[i] == predictedClasses[i]): correct += 1

    return correct / len(trueClass)

###############################################################################

def trainUnsup(df):
    '''
    initialise instances with random (non-uniform) class distributions
    '''
    classList = set(df.iloc[:,-1])  # extract unique classes
    classCount = len(classList)
    cleanTrain = df.drop(df.columns[-1], axis=1)  # drop class col
    N = cleanTrain.shape[0]  # instance count
    priorCounts = float()  #initialise prior
    randoms = []

    # initialise class probabilities as float
    for c in classList: cleanTrain[c] = float()

    # generate N random probability distributions, while summing for prior
    # store generated probabilities in dataframe
    for i in range(N):
        randoms.append(randDistGen(classCount))
        priorCounts += randoms[i]

        for idx, c in enumerate(classList):
            cleanTrain.at[i, c] = randoms[i][idx]

    # slide example
    # randoms2 = [[ 0.4,  0.6],
    #             [ 0.7,  0.3],
    #             [ 0.9,  0.1],
    #             [ 0.2,  0.8],
    #             [ 0.6,  0.4]]
    # randoms = randoms2
    # priorCounts= np.array([2.8, 2.2])

    # print('priorCounts', priorCounts)
    # print('priorProb', priorCounts / N)
    #print('INIT\n', cleanTrain)

    return cleanTrain, classList, priorCounts

###############################################################################

def predictUnsup(cleanTrain, classes, priorCounts):
    '''
    returns predicted classes and final class distributions
    '''
    N = cleanTrain.shape[0]  # instance count
    attribCount = cleanTrain.shape[1] - len(classes)
    priorProb = priorCounts / N

    for j in range(ITERATIONS):
        posteriorProb = defaultdict(lambda: 0)

        # generate attribute value|class pair probabilities
        for attrib in range(attribCount) :
            attribValues = list(cleanTrain[attrib].unique())
            if ('?' in attribValues): attribValues.remove('?')

            for idx, c in enumerate(classes):
                for val in attribValues:
                    key = createKey([attrib, val, c])
                    filterClassVal = cleanTrain[cleanTrain[attrib] == val]
                    posteriorProb[key] += filterClassVal[c].sum() / priorCounts[idx]

        # maximum likelihood estimation of each instance
        for i, instance in cleanTrain.iterrows():
            classSum = 0.0

            for idx, c in enumerate(classes):
                tmpProb = priorProb[idx]

                for attrib, val in enumerate(list(instance)):
                    key = createKey([attrib, val, c])
                    if key in posteriorProb: tmpProb *= posteriorProb[key]
                classSum += tmpProb
                cleanTrain.at[i, c] = tmpProb

            # normalise posterior
            for c in classes: cleanTrain.at[i, c] /= classSum

        # recalculate prior
        for idx,c in enumerate(classes): priorCounts[idx] = cleanTrain[c].sum()
        priorProb = priorCounts / N

    #print(cleanTrain)

    predictedClasses = []
    # extract final predictions (most likely class)
    for i, instance in cleanTrain.iterrows():
        currentMax = ['null', 0]

        for idx, c in enumerate(classes):
            if instance[c] > currentMax[1]: currentMax = [c, instance[c]]
        predictedClasses.append(currentMax[0])

    return predictedClasses, cleanTrain

###############################################################################

def evaluateUnsup(trueClass, predictedClasses, classes, flag):
    '''
    builds a confusion matrix for unsupervised evaluation
    '''
    if len(trueClass) != len(predictedClasses):
        print('Error: Class length')
        return

    # Create a pandas dataframe actual is the row, predicted is the column
    confusionMatrix = pd.DataFrame()
    for c in classes: confusionMatrix[c] = [0] * len(classes)

    confusionMatrix.index = classes  # index by classes

    # Calculate the confusion matrix
    for i in range(len(trueClass)):
        confusionMatrix.loc[trueClass[i], predictedClasses[i]] += 1

    # Add actual and predicted labels
    predictedCol = []
    actualRow = []

    for string in classes:
        predictedCol.append(string + ' (Predicted)')
        actualRow.append(string + ' (Actual)')

    confusionMatrix.columns = predictedCol
    confusionMatrix.index = actualRow

    if flag: print(confusionMatrix)

    # calculate unsupervised accuracy
    maxSum = 0
    totalSum = confusionMatrix.values.sum()
    # sum rows or columns???
    confusionMatrix = confusionMatrix.transpose()  # fix accuracy calc
    for c in confusionMatrix.columns: maxSum += confusionMatrix[c].max()

    return maxSum / totalSum

###############################################################################

def sample(func, desc):
    '''
    runs given function 10 times and takes average of measure
    '''
    RUNS = 10
    print(desc)  # description of experiment

    for d in DATASETS:
        avgMeasure = 0
        for i in range(RUNS): avgMeasure += func(d)
        print('{} | Avg. Measure: {}'.format(d, avgMeasure / RUNS))

###############################################################################

def mainQuestion3(data):
    '''
    test on training data for Question 3 (no holdout, supervised)
    '''
    df = pd.read_csv(data, header = None)
    priorProb, posteriorProb = trainSup(df)
    predictedClasses = predictSup(df, priorProb, posteriorProb)
    accuracy = evaluateSup(df, predictedClasses)
    #print('Dataset: {}, Accuracy: {}'.format(data, accuracy))

    return accuracy

###############################################################################

def deltaQuestion6(df, predict):
    '''
    Calculates how far away probabilistic estimate of true class is from 1.
    Assumes probabilistic estimate of true class = highest probability of all classes due to class 'swapping'
    '''
    deltaSum = 0

    # difference (probability) between each predicted class and 1
    for i, row in df.iterrows(): deltaSum += abs(1 - row[predict[i]])

    return deltaSum / df.shape[0]

###############################################################################

def mainSup(data):
    '''
    execute supervised NB across 'data'
    '''
    trainSet, testSet = preprocessSup(data)
    priorProb, posteriorProb = trainSup(trainSet)
    predictedClasses = predictSup(testSet, priorProb, posteriorProb)
    accuracy = evaluateSup(testSet, predictedClasses)
    #print('Dataset: {}, Accuracy: {}'.format(data, accuracy))

    return accuracy

###############################################################################

def mainUnsup(data):
    '''
    execute unsupervised NB across 'data'
    '''
    df = pd.read_csv(data, header = None)
    trueClass = df.iloc[:,-1].tolist()  # extract true classes
    cleanTrain, classes, priorCounts = trainUnsup(df)
    predictedClasses, finalDf = predictUnsup(cleanTrain, classes, priorCounts)
    accuracyUnsup = evaluateUnsup(trueClass, predictedClasses, classes, True)
    deltaAvg = deltaQuestion6(finalDf, predictedClasses)

    print('delta average', deltaAvg)

    return accuracyUnsup

###############################################################################

print(mainUnsup(DATASET3))

# sample(mainQuestion3, 'no holdout')
# sample(mainSup, 'with holdout')
# sample(mainUnsup, 'unsupervised delta testing')
# sample(mainUnsup, 'accuracy')
