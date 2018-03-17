import numpy as np
import pylab as plt
from lr_utils import load_dataset


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def initialize_with_zeros(dim):
    w = np.zeros([dim, 1])
    b = 0
    return w, b


def propagate(w, b, X, Y):
    m = X.shape[1]

    ## FORWARD PROPAGATION
    A = sigmoid(np.dot(w.T, X) + b)
    cost = (-1 / m) * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))

    ## BACKWARD PROPAGATION
    dw = (1 / m) * np.dot(X, (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    cost = np.squeeze(cost)

    return dw, db, cost


def optimize(w, b, X, Y, times, rate, print_cost=False):
    costs = []
    for i in range(times):
        dw, db, cost = propagate(w, b, X, Y)
        w -= rate * dw
        b -= rate * db

        if i % 100 == 0:
            costs.append([i / 100, cost])

        if print_cost and i % 100 == 0:
            print('Cost :', i, 'times:', cost)

    params = {'w': w, 'b': b}

    grads = {'dw': dw, 'db': db}

    return params, grads, costs


def accuracy(w, b, X, data):
    m = X.shape[1]
    Y_prediction = np.zeros([1, m])
    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(m):
        Y_prediction[0][i] = 1 if A[0, i] > 0.5 else 0

    return 100 - np.mean(np.abs(Y_prediction - data)) * 100


def showCost(costs, rate):
    plt.xlabel('iterations (per hundreds)')
    plt.ylabel('cost')
    plt.title('Learning rate = ' + str(rate))
    for i in range(len(costs)):
        plt.plot(costs[i][0], costs[i][1], 'ob')
    plt.show()


def saveResults(w, b):
    res = np.vstack([b, w])
    np.savetxt('iscat.txt', res, delimiter=',')


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# 训练集中样本个数
m_train = train_set_y.shape[1]
# 测试集中样本个数
m_test = test_set_y.shape[1]
# 每个图片为num_px*num_px的
num_px = train_set_x_orig.shape[1]

train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255

dim = num_px ** 2 * 3
w, b = initialize_with_zeros(dim)
rate = 0.005
times = 950
X = train_set_x
Y = train_set_y

params, grads, costs = optimize(w, b, X, Y, times, rate, True)

w = params['w']
b = params['b']

print("Times:", times)
print("Train Accuracy:", accuracy(w, b, X, train_set_y))
print("Test Accuracy:", accuracy(w, b, test_set_x, test_set_y))

print(w, b)
saveResults(w, b)

showCost(costs, rate)
