# Chirag Rao Sahib      : 836011
# Maleakhi Agung Wijaya : 784091
# Date                  : 28/03/2018
# Python version        : 3.6.4

###############################################################################

import numpy as np
import pandas as pd
from collections import defaultdict

SPLIT_RATIO = 0.8  # holdout ratio (according to Pareto principle)
ITERATIONS = 10  # iterations for unsupervised NB
EPSILON = 10**(-6)  # value for epsilon smoothing

DATASET1 = '2018S1-proj1_data/breast-cancer-dos.csv'
DATASET2 = '2018S1-proj1_data/car-dos.csv'
DATASET3 = '2018S1-proj1_data/hypothyroid-dos.csv'
DATASET4 = '2018S1-proj1_data/mushroom-dos.csv'
DATASETS = [DATASET1, DATASET2, DATASET3, DATASET4]
SAMPLE = '2018S1-proj1_data/sample.csv'  # example dataset from lecture notes

###############################################################################

'''
Helper function to create dictionary key given a string
@param lst = list of string to combined
@return dictionary key for probability (i.e. 0|mild|flu = (0==mild given flu))
'''
def createKey(lst):
    return '|'.join(str(i) for i in lst)

###############################################################################

'''
Random generator
@param length = length of the random array
@return array containing random numbers that sums to 1
'''
def randDistGen(length):
    dist = np.random.random(length)
    return dist / dist.sum()

###############################################################################

'''
Preprocessing for supervised to split the data into training/test
@param data = dataset
@param flag = True = no split, False = split
'''
def preprocessSup(data, flag = False):
    dataFrame = pd.read_csv(data, header = None)

    if (flag == False):
        # Split according to the split ratio
        split = np.random.rand(len(dataFrame)) < SPLIT_RATIO
        train = dataFrame[split]
        test = dataFrame[~split]
    else:
        train = dataFrame
        test = dataFrame

    return train, test

###############################################################################

'''
Create supervised Naive Bayes model by returning prior and posterior probability
@param trainSet = data that are used for training to generate model
@return priobProb, posteriorProb = probability counters
'''
def trainSup(trainSet):
    priorCounts = trainSet.iloc[:,-1].value_counts()
    priorProb = priorCounts / trainSet.shape[0]
    attribCount = trainSet.shape[1]
    posteriorProb = {}

    # Iterating over all columns except for the class column
    for attrib in range(attribCount - 1):
        # Generate list of unique attribute values and disregard ?
        attribValues = list(trainSet[attrib].unique())
        if ('?' in attribValues): attribValues.remove('?')

        # Calculate posterior probabilities
        for c in priorCounts.index:
            for val in attribValues:
                # first filter by class then by attribute value
                filterClass = trainSet[trainSet.iloc[:,-1] == c]
                filterClassVal = filterClass[filterClass[attrib] == val]

                # Generate key for dictionary (0|severe|flu means (column 0=severe|flu)
                key = createKey([attrib, val, c])
                posteriorProb[key] = filterClassVal.shape[0] / priorCounts[c]

    # Iterate probabilities to perform epsilon smoothing
    for key, value in posteriorProb.items():
        if (value == 0):
            posteriorProb[key] = EPSILON

    return priorProb, posteriorProb

###############################################################################

'''
Generate prediction for the testSet
@param testSet = data that will be classified
@param priorProb, posteriorProb = model
@return predictedClasses = array containing model predictions
'''
def predictSup(testSet, priorProb, posteriorProb):
    cleanTest = testSet.drop(testSet.columns[-1], axis=1)
    predictedClasses = []

    for i, instance in cleanTest.iterrows():
        currentMax = ['null', -float("inf")]  # track most probable class

        for c in priorProb.index:
            # maximum likelihood estimation of each instance
            prob = np.log(priorProb[c])

            for attrib, val in enumerate(list(instance)):
                key = createKey([attrib, val, c])
                if key in posteriorProb: prob += np.log(posteriorProb[key])

            if prob > currentMax[1]: currentMax = [c, prob]

        # predicted class = most likely class
        predictedClasses.append(currentMax[0])

    return predictedClasses

###############################################################################

'''
Simple accuracy measure of the supervised context
@param testSet = array of test result
@param predictedClasses = array of predicted result
@return accuracy = (TP+TN) / (TP+TN+FP+FN)
'''
def evaluateSup(testSet, predictedClasses):
    correct = 0
    trueClass = testSet.iloc[:,-1].tolist()

    if len(trueClass) != len(predictedClasses):
        print('Error: Class length')
        return

    for i in range(len(trueClass)):
        if (trueClass[i] == predictedClasses[i]): correct += 1

    return correct / len(trueClass)

###############################################################################

'''
Create confusion matrix for supervised and unsupervised
@param trueClass = actual class result array
@param predictedClasses = predicted classes array
@param classes = list of unique classes
@return confusionMatrix = confusion matrix
'''
def createConfusionMatrix(trueClass, predictedClasses, classes):
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

    return(confusionMatrix)

###############################################################################

'''
Preprocessing for unsupervised
@param data = dataset
@return dataFrame = pandas dataFrame consisting of data from the csv file
@return unsupervisedDataFrame = pandas dataFrame with class eliminated and probability added
'''
def preprocessUnsup(data):
    dataFrame = pd.read_csv(data, header = None)
    unsupervisedDataFrame = initialiseUnsup(dataFrame)

    return (dataFrame, unsupervisedDataFrame)

###############################################################################

'''
Initialise the dataset with random distribution
@param dataset = dataframe of the dataset
@return unsupervisedDataset = dataset initialised with random distribution
'''
def initialiseUnsup(dataset):
        rowNumber = dataset.shape[0]
        classColumn = list(dataset[dataset.shape[1] - 1].unique())
        classNumber = len(classColumn)
        unsupervisedDataset = dataset.drop(dataset.shape[1] - 1, axis=1) # drop the last column

        # sample randomly
        sampleMatrix = np.zeros((rowNumber, classNumber))
        for i in range(rowNumber):
            samples = randDistGen(classNumber)
            sampleMatrix[i] = samples

        # Add a column to the dataset according to random distribution (initialisation phase)
        rowInstance = unsupervisedDataset.shape[0]
        for c in classColumn:
            unsupervisedDataset[c] = [0 for i in range(rowInstance)]

        matrixCounter = 0
        # Iterate through the matrix and assign to the dataframe
        for index, row in unsupervisedDataset.iterrows():
            unsupervisedDataset.loc[index, -classNumber:] = sampleMatrix[matrixCounter]
            matrixCounter += 1

        return(unsupervisedDataset)

###############################################################################

'''
This function should build an unsupervised NB model and return a dictionary of prior and posterior probability
@param classColumn = possible class names (weak unsupervised model)
@param dataset = data that are used to create the unsupervised NB classifier (format after running initialiseUnsup function)
@return priorProb = dictionary describing prior probability of the class in training data
@return posteriorProb = dictionary of dictionaries describing posterior probability
'''
def trainUnsup(classColumn, dataset):
    classCount = len(classColumn)
    attribColumn = list(range(dataset.shape[1] - classCount))

    # Create a dictionary of prior probability
    priorProb = {}
    for c in classColumn:
        priorProb[c] = dataset[c].sum()

    # make prior to a probability
    total_sum = sum(priorProb.values())
    for c in classColumn:
        priorProb[c] /= total_sum

    # Create posterior
    posteriorProb = {}

    # Setup the dictionary component
    for col in attribColumn:
        posteriorProb[col] = {}
        for c in classColumn:
            posteriorProb[col][c] = {}
            for uc in dataset[col].unique():
                posteriorProb[col][c][uc] = 0

    # Now use the training data to perform count calculation
    for index, row in dataset.iterrows():
        for col in attribColumn:
            for c in classColumn:
                posteriorProb[col][c][row[col]] += row[c]

    # After we finish the count calculation, perform probability calculation
    # Now calculate the posterior probability
    for col in attribColumn:
        for c in classColumn:
            sumInstance = sum(posteriorProb[col][c].values())
            for uc in dataset[col].unique():
                posteriorProb[col][c][uc] /= sumInstance

                # Perform epsilon smoothing
                if (posteriorProb[col][c][uc] == 0):
                    posteriorProb[col][c][uc] = EPSILON

    # Return the dictionary of prior probability and posterior probability
    return (priorProb, posteriorProb)

###############################################################################

'''
This function should predict the class for a set of instances, based on a trained model
@param classColumn = possible class name (weak unsupervised model)
@param dataset = data that are used to calculate prediction
@param priorProb = dictionary of probability counter
@param posteriorProb = dictionary of probability counter
@return testClass = the class predicted by the naive bayes classifier.
    The predict class will change the structure of dataset to be used for the next iteration.
'''
def predictUnsup(classColumn, dataset, priorProb, posteriorProb):
    classCount = len(classColumn)
    testClass = [] # used to capture test result
    attribColumn = list(range(dataset.shape[1] - classCount))

    # Get the answer for every test instance
    for index, row in dataset.iterrows():
        # Initiate dictionary capturing the values calculated by naive bayes model
        testValue = {}
        for c in classColumn:
            testValue[c] = 0

        # Calculate for each class using the naive bayes model (log model for multiplication)
        for c in classColumn:
            testValue[c] = np.log(priorProb[c])
            for col in attribColumn:
                testValue[c] += np.log(posteriorProb[col][c][row[col]])

        # After calculating all of the possible class, we want to choose the maximum
        maximumClass = classColumn[0]
        maximumValue = testValue[maximumClass]
        for key, value in testValue.items():
            if (value > maximumValue):
                maximumValue = value
                maximumClass = key

        # Append result to be returned
        testClass.append(maximumClass)

        # Change the dataset structure for the instance to prepare for the next iteration
        # First take the exponent of that to get the real probability calculation value
        for c in classColumn:
            testValue[c] = np.exp(testValue[c])

        # Calculate the new probability
        denominatorNew = sum(testValue.values())

        for c in classColumn:
            dataset.loc[index, c] = testValue[c] / denominatorNew

    # Return the classifier for the class
    return testClass

###############################################################################

'''
Used to calculate the accuracy of the unsupervised model
@param confusionMatrix = confusion matrix of the unsupervised result
@result accuracy of the unsupervised taking into account class swapping
'''
def evaluateUnsup(confusionMatrix):
    maxSum = 0
    totalSum = confusionMatrix.values.sum()

    # Calculate sum of the highest of each column as per forum thread:
        # Swapping in Evaluation of Unsupervised NB Classifier
    for c in confusionMatrix.columns: maxSum += confusionMatrix[c].max()

    return (maxSum/totalSum)

###############################################################################

'''
Used mainly in holdout method to average 10 executions
@param func = function that will be run
@param desc = description of experiment
@param flag = if true dont split else split data (default = split)
@param flagPrint = true print, false otherwise
'''
def sampleExperiment(func, desc, flag, flagPrint):
    RUNS = 10
    print(desc)  # description of experiment

    for d in DATASETS:
        avgMeasure = 0
        for i in range(RUNS): avgMeasure += func(d, flag, flagPrint)
        print('{} | Avg. Measure: {}'.format(d, avgMeasure / RUNS))

###############################################################################

'''
Main function for supervised to be run across a dataset
@param data = dataset used to run
@param flag = if true dont split else split data (default = split)
@param flagPrint = if true print confusion matrix
@return accuracy = accuracy of the data
'''
def mainSup(data, flag=False, flagPrint=True):
    # If true (don't split), false split
    trainSet, testSet = preprocessSup(data, flag)

    priorProb, posteriorProb = trainSup(trainSet)
    predictedClasses = predictSup(testSet, priorProb, posteriorProb)
    accuracy = evaluateSup(testSet, predictedClasses)
    confusionMatrix = createConfusionMatrix(testSet.iloc[:,-1].tolist(), predictedClasses, testSet.iloc[:, -1].unique())

    if (flagPrint):
        print(confusionMatrix)
        print("\nThe accuracy for the dataset is {}.\n\n".format(accuracy))

    # Return accuracy
    return accuracy

###############################################################################

'''
Used to answer question 3, comparing holdout and training on test data
'''
def mainQuestion3():
    # Using holdout
    sampleExperiment(mainSup, "Using holdout, averaged over 10 runs", False, False)

    print("\n")

    #no holdout
    sampleExperiment(mainSup, "Training in test data", True, False)

###############################################################################

'''
Calculates how far away probabilistic estimate of true class is from 1.
Assumes probabilistic estimate of true class = highest probability of all classes due to class 'swapping'
@param df = dataframe containing the data with the probability columns
@param predict = prediction (highest value probability, see our assumption above)
@return average = delta average
'''
def deltaQuestion6(df, predict):
    deltaSum = 0

    # difference (probability) between each predicted class and 1
    for i, row in df.iterrows(): deltaSum += abs(1 - row[predict[i]])

    return deltaSum / df.shape[0]

###############################################################################

'''
Execute unsupervised, will print the confusion matrix
@param data = number of data
@param iteration = number of iteration
@return accuracy = accuracy based on evaluateUnsup
'''
def mainUnsup(data, iteration):
    datas = preprocessUnsup(data)
    df = datas[0]
    unsupervisedDataFrame = datas[1]
    trueClass = df.iloc[:,-1].tolist()  # extract true classes

    # Iterate iteration number of times changing the unsupervisedDataFrame
    for i in range(iteration):
        print("Iteration {}".format(i+1))
        prior, posterior = trainUnsup(list(set(trueClass)), unsupervisedDataFrame)
        oldUnsupervisedDf = unsupervisedDataFrame.copy(deep=True)
        predictedClasses = predictUnsup(list(set(trueClass)), unsupervisedDataFrame, prior, posterior)
        confusionMatrix = createConfusionMatrix(trueClass, predictedClasses, list(set(trueClass)))
        accuracyUnsup = evaluateUnsup(confusionMatrix)
        print(confusionMatrix)
        print("The accuracy is {}.".format(accuracyUnsup))

        # Delta
        deltaAverage = deltaQuestion6(oldUnsupervisedDf, predictedClasses)
        print("The delta average is {}".format(deltaAverage))
        print("\n\n")

    return accuracyUnsup

###############################################################################

mainSup(DATASET3, False, True)

mainUnsup(DATASET3, 10)
