import numpy as np
import pylab


# 计算每次拟合的m,b的误差
def error(b, m, data):
    x = data[:, 0]
    y = data[:, 1]
    totalError = (y - m * x - b) ** 2
    totalError = np.sum(totalError, axis=0)

    return totalError / len(data)


def optimizer(data, b, m, learning_rate, times):
    # gradient descent
    for i in range(times):
        [b, m] = compute_gradient(b, m, data, learning_rate)

        if i % 100 == 0:
            print('iter {0}:error={1}'.format(i, error(b, m, data)))
    return [b, m]


def compute_gradient(b_current, m_current, data, learning_rate):
    N = len(data)
    x = data[:, 0]
    y = data[:, 1]

    b_gradient = -(2 / N) * (y - m_current * x - b_current)
    b_gradient = np.sum(b_gradient, axis=0)

    m_gradient = -(2 / N) * x * (y - m_current * x - b_current)
    m_gradient = np.sum(m_gradient, axis=0)

    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def plot_data(data, b, m):
    x = data[:, 0]
    y = data[:, 1]
    y_predict = m * x + b
    pylab.plot(x, y, 'o')
    pylab.plot(x, y_predict, 'k-')
    pylab.show()


def Linear_regression():
    # get train data
    data = np.loadtxt('linear_regression.txt', delimiter=',')

    learning_rate = 0.002
    initial_b = 0.0
    initial_m = 0.0
    num_iter = 25000

    # print b m error
    print('initial variables:\n initial_b = {0}\n intial_m = {1}\n error of begin = {2} \n' \
          .format(initial_b, initial_m, error(initial_b, initial_m, data)))

    # optimizing b and m
    [b, m] = optimizer(data, initial_b, initial_m, learning_rate, num_iter)

    # print final b m error
    print(
        'final formula parmaters:\n b = {1}\n m={2}\n error of end = {3} \n'.format(num_iter, b, m, error(b, m, data)))

    # plot result
    plot_data(data, b, m)


Linear_regression()
