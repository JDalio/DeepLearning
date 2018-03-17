import numpy as np
import pylab as plt


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def loadData():
    data = np.loadtxt('dataset/logistic_regression.txt', delimiter=',')
    train_x = data[..., 0:2]
    train_y = data[..., 2]
    train_y = train_y.reshape([train_y.shape[0], 1])
    return [train_x.T, train_y.T]


def trainLogRegres(train_x, train_y, opts):
    alpha = opts['alpha']
    maxIter = opts['maxIter']
    weights = np.zeros([train_x.shape[0], 1])
    bias = 0.0
    print('Init :\nWeights :\n', weights, '\nBias :', bias)
    for i in range(maxIter):
        a = sigmoid(np.dot(weights.T, train_x))
        error = train_y - a

        bias += alpha * np.mean(error)
        weights += alpha * np.dot(train_x, error.T)
    return weights, bias


def testLogRegres(res, test_x, test_y):
    featureNum, sampleNum = np.shape(test_x)
    matchCount = 0
    for i in range(sampleNum):
        predict = sigmoid(np.dot(res[0].T, test_x[..., i]) + res[1]) > 0.5
        if (predict == bool(test_y[..., i])):
            matchCount += 1
    return float(matchCount) / sampleNum


def showLogRegres(res, train_x, train_y):
    featureNum, sampleNum = np.shape(train_x)
    print(sampleNum)
    for i in range(sampleNum):
        if train_y[0, i] == 0:
            plt.plot(train_x[..., i][0], train_x[..., i][1], 'or')
        else:
            plt.plot(train_x[..., i][0], train_x[..., i][1], 'ob')

    min_x = np.min(train_x[0])
    max_x = np.max(train_x[0])
    a1, a2 = res[0]
    b = res[1]
    min_y = -(a1 * min_x + b) / a2
    max_y = -(a1 * max_x + b) / a2

    plt.plot([min_x, max_x], [min_y, max_y], '-g')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


## step 1: load data
train_x, train_y = loadData()
# print(np.dot(np.array([1,3]).reshape([1,2]),train_x[...,0]).shape)

## step 2： training
opts = {'alpha': 0.001, 'maxIter': 4000, 'optimizeType': 'GradDescent'}
res = trainLogRegres(train_x, train_y, opts)

## step 3: testing
accuracy = testLogRegres(res, train_x, train_y)
# print(accuracy)

## step 4: show the result
print('Result:\n', 'Weights:\n', res[0], '\nBias:', res[1])
print('Accuracy:', accuracy)
showLogRegres(res, train_x, train_y)
