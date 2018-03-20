import numpy as np
import pylab as plt
import sklearn
import sklearn
import sklearn.datasets
import sklearn.linear_model
from testCases import *
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets


# clf = sklearn.linear_model.LogisticRegressionCV()
# clf.fit(X.T, np.squeeze(Y))
# plot_decision_boundary(lambda x: clf.predict(x), X, Y)
# plt.title('Logistic Regression')
# predict = np.array(clf.predict(X.T), dtype='i4')
# Y = np.array(Y.ravel(), dtype='i4')
# print(accuracy)

def compute_accuracy(predict, Y):
    accuracy = 100 - np.mean(np.abs(predict - Y)) * 100
    return accuracy


def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_y = Y.shape[0]
    return n_x, n_y


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


def backword_propagation(parameters, cache, X, Y):
    m = Y.shape[1]

    Z1 = cache['Z1']
    A1 = cache['A1']
    Z2 = cache['Z2']
    A2 = cache['A2']
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dA2 = (-1 / m) * (Y / A2 - (1 - Y) / (1 - A2))
    dZ2 = (-1 / m) * (Y - A2)

    dW2 = np.dot(A1, dZ2.T)
    db2 = np.mean(dZ2)

    dA1 = np.dot(W2, dZ2)
    dZ1 = (1 - np.tanh(Z1) ** 2) * dA1

    dW1 = np.dot(X, dZ1.T)
    db1 = np.mean(dZ1, 1)[:, np.newaxis]

    grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}

    return grads


def updata_parameters(parameters, grads, rate=1.2):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 -= rate * dW1
    b1 -= rate * db1
    W2 -= rate * dW2
    b2 -= rate * db2

    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}

    return parameters


def nn_model(X, Y, n_h, rate, num_iterations, print_cost=False):
    np.random.seed(3)
    n_x, n_y = layer_sizes(X, Y)
    parameters = initialize_parameters(n_x, n_h, n_y)
    for i in range(num_iterations):
        A2, cache = forward_propagation(X, parameters)
        cost = compute_cost(A2, Y)
        grads = backword_propagation(parameters, cache, X, Y)
        parameters = updata_parameters(parameters, grads, rate)

        if print_cost and i % 50 == 0:
            print(i, 'Times:', cost)
    return parameters


def predict(parameters, X):
    A2, cache = forward_propagation(X, parameters)
    predictions = np.round(A2)
    return predictions


# np.random.seed(1)
# X, Y = load_planar_dataset()

np.random.seed(1)
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

X, Y = datasets["noisy_moons"]
X, Y = X.T, Y.reshape(1, Y.shape[0])

# try different hidden layers
plt.figure(figsize=(16, 32))
hidden_layer_size = [1, 5, 10, 20, 30, 50]
for i, n_h in enumerate(hidden_layer_size):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden layer of size ' + str(n_h))

    parameters = nn_model(X, Y, n_h, 1.2, 10000, False)
    predictions = predict(parameters, X)
    accuracy = compute_accuracy(predictions, Y)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    print('Accuracy for {} hidden units:{}%'.format(n_h, accuracy))
plt.show()
