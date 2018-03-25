import numpy as np
import pandas as pd
from collections import defaultdict

#PARETO PRINCIPLE
SPLIT_RATIO = 0.8
ITERATIONS = 4

DATASET1 = '2018S1-proj1_data/breast-cancer-dos.csv'
DATASET2 = '2018S1-proj1_data/car-dos.csv'
DATASET3 = '2018S1-proj1_data/hypothyroid-dos.csv'
DATASET4 = '2018S1-proj1_data/mushroom-dos.csv'
DATASETS = [DATASET1, DATASET2, DATASET3, DATASET4]
SAMPLE = '2018S1-proj1_data/sample.csv'



def createKey(lst):
    return '|'.join(str(i) for i in lst)

def randDistGen(length):
    dist = np.random.random(length)
    return dist / dist.sum()


def preprocess(dataset):
    dataFrame = pd.read_csv(dataset, header = None)
    split = np.random.rand(len(dataFrame)) < SPLIT_RATIO
    train = dataFrame[split]
    test = dataFrame[~split]
    return train, test


def train(trainSet):
    priorCounts = trainSet.iloc[:,-1].value_counts()
    priorProb = priorCounts / trainSet.shape[0]
    attribCount = trainSet.shape[1]
    posteriorProb = {}

    for attrib in range(attribCount-1) :

        attribValues = list(trainSet[attrib].unique())
        if ('?' in attribValues): attribValues.remove('?')

        for cls in priorCounts.index:
            for val in attribValues:
                filterClass = trainSet[trainSet.iloc[:,-1] == cls]
                filterClassVal = filterClass[filterClass[attrib] == val]
                key = createKey([attrib, val, cls])
                posteriorProb[key] = filterClassVal.shape[0] / priorCounts[cls]

    return priorProb, posteriorProb



def predict(test, priorProb, posteriorProb):
    cleanTest = test.drop(test.columns[-1], axis=1)
    predictedClasses = []

    for i, instance in cleanTest.iterrows():
        currentMax = ['null', 0]

        for cls in priorProb.index:
            prob = priorProb[cls]

            for attrib, val in enumerate(list(instance)):
                key = createKey([attrib, val, cls])
                if key in posteriorProb: prob *= posteriorProb[key]

            if prob > currentMax[1]:
                currentMax = [cls, prob]

        predictedClasses.append(currentMax[0])

    return predictedClasses


def evaluate(testSet, predictedClasses):
    correct = 0
    classes = testSet.iloc[:,-1].tolist()

    if (len(classes) != len(predictedClasses)): print('Error: class length')

    for i in range(len(classes)):
        if (classes[i] == predictedClasses[i]): correct += 1

    return correct / len(classes)



def main_supervised(data):
    trainSet, testSet = preprocess(data)
    priorProb, posteriorProb = train(trainSet)
    predictedClasses = predict(testSet, priorProb, posteriorProb)
    accuracy = evaluate(testSet, predictedClasses)
    #print('Dataset: {}, Accuracy: {}'.format(data, accuracy))
    return accuracy



def main_unsupervised(data):
    dataFrame = pd.read_csv(data, header = None)
    cleanTrain, classes, priorCounts = train_unsupervised(dataFrame)
    predictedClasses, finalDf = predict_unsupervised(cleanTrain, classes, priorCounts)
    trueClass = dataFrame.iloc[:,-1].tolist()
    confusionMatrix = evaluate_unsupervised(trueClass, predictedClasses, classes)
    #print(confusionMatrix)
    sum1 = confusionMatrix.values.sum()
    sum2 = 0
    for c in confusionMatrix.columns: sum2 += confusionMatrix[c].max()
    accuracy = sum2/sum1

   # print('Confusion Accuracy', accuracy)

    # q6 delta
    deltaSum = 0
    for i, instance in finalDf.iterrows():
        # probabilistic estimate of true class = highest probability of all classes
        # due to class swaping
        deltaSum += abs(1- instance[predictedClasses[i]])

    deltaSum /= finalDf.shape[0]
    # print('iterations', ITERATIONS)
    print('delta average Q6', deltaSum)

    return accuracy


def train_unsupervised(df):
    classes = set(df.iloc[:,-1])
    classCount = len(classes)
    cleanTrain = df.drop(df.columns[-1], axis=1)
    N = cleanTrain.shape[0]
    priorCounts = float()
    randoms = []

    for i in range(N):
        randoms.append(randDistGen(classCount))
        priorCounts += randoms[i]

    # force feed (slide example)
    # randoms2 = [[ 0.4,  0.6],
    #             [ 0.7,  0.3],
    #             [ 0.9,  0.1],
    #             [ 0.2,  0.8],
    #             [ 0.6,  0.4]]
    # randoms = randoms2
    # priorCounts= np.array([2.8, 2.2])

    # print('priorCounts', priorCounts)
    # print('priorProb', priorCounts / N)

    for c in classes: cleanTrain[c] = float()

    for i in range(N):
        for idx, c in enumerate(classes):
            cleanTrain.at[i, c] = randoms[i][idx]

    #print('INIT\n', cleanTrain)
    return cleanTrain, classes, priorCounts



def predict_unsupervised(cleanTrain, classes, priorCounts):
    attribCount = cleanTrain.shape[1] - len(classes)
    N = cleanTrain.shape[0]
    priorProb = priorCounts / N

    for j in range(ITERATIONS):
        posteriorProb = defaultdict(lambda: 0)

        for attrib in range(attribCount) :

            attribValues = list(cleanTrain[attrib].unique())
            if ('?' in attribValues): attribValues.remove('?')

            for idx, c in enumerate(classes):
                for val in attribValues:
                    key = createKey([attrib, val, c])
                    filterClassVal = cleanTrain[cleanTrain[attrib] == val]
                    posteriorProb[key] += filterClassVal[c].sum() / priorCounts[idx]

        for i, instance in cleanTrain.iterrows():
            classSum = 0.0
            for idx, c in enumerate(classes):
                tmpProb = priorProb[idx]

                for attrib, val in enumerate(list(instance)):
                    key = createKey([attrib, val, c])
                    if key in posteriorProb: tmpProb *= posteriorProb[key]

                classSum += tmpProb
                cleanTrain.at[i, c] = tmpProb

            #normalise
            for c in classes:
                cleanTrain.at[i, c] = cleanTrain.at[i, c] / classSum

        #recalculate prior
        for idx,c in enumerate(classes): priorCounts[idx] = cleanTrain[c].sum()

        priorProb = priorCounts / N

    #print(cleanTrain)

    predictedClasses = []

    for i, instance in cleanTrain.iterrows():
        currentMax = ['null', 0]

        for idx, c in enumerate(classes):
            if instance[c] > currentMax[1]: currentMax = [c, instance[c]]

        predictedClasses.append(currentMax[0])

    return predictedClasses, cleanTrain



def evaluate_unsupervised(true_test_result, predicted_test_result, class_column):

    if (len(true_test_result) != len(predicted_test_result)):
        print("Error, different length.")

    else:
        # Create a pandas dataframe actual is the row, predicted is the column
        confusion_df = pd.DataFrame()

        for unique_class in class_column:
            confusion_df[unique_class] = [0 for i in range(len(class_column))]

        # Change index for df
        confusion_df.index = class_column

        # Calculate the confusion matrix
        for i in range(len(true_test_result)):
            confusion_df.loc[true_test_result[i], predicted_test_result[i]] += 1

        # Add actual and predicted description on the table to make it easier to see
        predicted_column = []
        for string in confusion_df.columns:
            string += " predicted"
            predicted_column.append(string.title())

        actual_row = []
        for string in class_column:
            string += " actual"
            actual_row.append(string.title())

        confusion_df.columns = predicted_column
        confusion_df.index = actual_row

        return confusion_df



def sample(main, str):
    print(str)
    for d in DATASETS:
        avg_acc = 0
        for i in range(10):
            avg_acc += main(d)
        print('Dataset: {}, Accuracy: {}'.format(d, avg_acc/10))


def main_q3(data):
    '''
    test on training data
    '''
    df = pd.read_csv(data, header = None)
    priorProb, posteriorProb = train(df)
    predictedClasses = predict(df, priorProb, posteriorProb)
    accuracy = evaluate(df, predictedClasses)
    #print('Dataset: {}, Accuracy: {}'.format(data, accuracy))
    return accuracy






# sample(main_q3, 'no holdout')
# sample(main_supervised, 'with holdout')
# sample(main_unsupervised, 'unsupervised delta testing')

# sample(main_unsupervised, 'accuracy')


main_unsupervised(DATASET2)
