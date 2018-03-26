import numpy as np
import h5py
import pylab as  plt

# %matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# np.pad(),padding an image
# four params:上下左右
a = np.array([[1, 1], [2, 2]], dtype='f4')
b = np.pad(a, ((1, 1), (1, 1)), 'constant')


# print(b)

def zero_pad(X, pad):
    '''
    :param X: (m,n_H,n_W,n_C)
    :param pad: padding num
    :return: (m,n_H+2*pad,n_W+2*pad,n_C)
    '''
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
    return X_pad


x = np.random.randn(1, 3, 3, 3)
print(x)
print(x.round(0))
x = zero_pad(x, 2)
