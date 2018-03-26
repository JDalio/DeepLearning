import numpy as np


def zero_pad(X, pad):
    '''
    :param X: (m,n_H,n_W,n_C)
    :param pad: padding num
    :return: (m,n_H+2*pad,n_W+2*pad,n_C)
    '''
    X_pad = np.pad(X, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant')
    return X_pad


def conv_single_step(a_slice_prev, W, b):
    '''
    :param a_slice_prev: one slide window(f,f,n_C_pre) in original img
    :param W: filter(f,f,n_C_pre)
    :param b: bias(1,1,1)
    :return: the result of convoluing a slide window
    '''
    s = a_slice_prev * W
    Z = np.sum(s)
    return Z + float(b)


def conv_forward(A_prev, W, b, hparameters):
    '''
    The forward propagation for a convolutional function
    :param A_prev:(m,n_H_prev,n_W_prev,n_C_prev)
    :param W: (f,f,n_C_prev,n_C),the primitive channal of image is n_C_prev,the number of filters is n_C
    :param b: (1,1,1,n_C)
    :param hparameters: stride and pad
    :return:(m,n_H,n_W,n_C)
    '''
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape()
    (f, f, n_C_prev, n_C) = W.shape
    stride = hparameters['stride']
    pad = hparameters['pad']

    n_H = int((n_H_prev + 2 * pad - f) / stride + 1)
    n_W = int((n_W_prev + 2 * pad - f) / stride + 1)

    Z = np.zeros(m, n_H, n_W, n_C)

    A_prev_pad = zero_pad(A_prev, pad)

    for i in range(m):  # total m images
        a_prev_pad = A_prev_pad[i, :]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    Z[i, h, w, c] = conv_single_step(a_slice_prev, W[:, :, :, c], b[:, :, :, c])
    return Z


def pool_forward(A_prev, hparameters, mode='max'):
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    f = hparameters['f']
    stride = hparameters['stride']

    n_H = int(1 + (n_H_prev - 2 * f) / stride)
    n_W = int(1 + (n_W_prev - 2 * f) / stride)
    n_C = n_C_prev

    A = np.zeros(m, n_H, n_W, n_C)

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    vert_start = h * stride
                    vert_end = vert_start + f
                    horiz_start = w * stride
                    horiz_end = horiz_start + f

                    a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]

                    if mode == 'max':
                        A[i, h, w, c] = np.max(a_prev_slice)
                    elif mode == 'average':
                        A[i, h, w, c] = np.mean(a_prev_slice)

    return A
