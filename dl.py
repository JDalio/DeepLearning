import numpy as np


def optimizer(data, b, m, rate, times):
    for i in range(times):
        [b, m] = compute_gradient(b, m, data, rate)
        print(i, 'times:', [b, m])
    return [b, m]


def compute_gradient(b_cur, m_cur, data, rate):
    b_gra = 0
    m_gra = 0

    N = float(len(data))

    for i in range(len(data)):
        x = data[i, 0]
        y = data[i, 1]

        b_gra += -(2 / N) * (y - (m_cur * x + b_cur))
        m_gra += -(2 / N) * x * (y - (m_cur * x + b_cur))

    new_b = b_cur - rate * b_gra
    new_m = m_cur - rate * m_gra

    return [new_b, new_m]


data = np.loadtxt('./dataset/linear_regression.txt', delimiter=',')

rate = 0.01
b = 0.0
m = 0.0
times = 1000

[b, m] = optimizer(data, b, m, rate, times)
