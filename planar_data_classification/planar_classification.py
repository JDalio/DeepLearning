import numpy as np
import pylab as plt
import sklearn
import sklearn
import sklearn.datasets
import sklearn.linear_model
from testCases import *
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)
X, Y = load_planar_dataset()

# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T, np.squeeze(Y))
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# plt.title('Logistic Regression')
# predict = np.array(clf.predict(X.T), dtype='i4')
# Y = np.array(Y.ravel(), dtype='i4')
# accuracy = 100 - np.mean(np.abs(predict - Y)) * 100
# print(accuracy)

X_assess, Y_assess = layer_sizes_test_case()


def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    W1 = np.random.randn(n_x, n_h) * 0.01
    b1 = np.zeros([n_h, 1])
    W2 = np.random.randn(n_h, n_y)
    b2 = np.zeros([n_y, 1])

    parameters = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    return parameters


def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    Z1 = np.dot(W1.T, X) + b1
    A1 = np.tanh(Z1)

    Z2 = np.dot(W2.T, A1) + b2
    A2 = sigmoid(Z2)

    cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2}

    return A2, cache


def compute_cost(A2, Y):
    m = Y.shape[1]
    cost = (-1 / m) * np.sum(Y * np.log(A2) + (1 - Y) * np.log(1 - A2))
    return np.squeeze(cost)
