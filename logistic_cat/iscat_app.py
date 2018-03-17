import numpy as np
import cv2


# Test a image if there is a cat
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


key = np.loadtxt('iscat.txt', delimiter=',')
b = key[0]
w = key[1:]
w = w.reshape([w.shape[0], 1])

path = input('Input the path of your iamge(64x64):\n')
img = cv2.imread(path)
x = img.reshape([-1, 1])
a = np.dot(w.T, x) + b
srate = np.squeeze(sigmoid(a))
if srate > 0.5:
    print('Have a cat')
    print("Similarity Rate:", srate)
else:
    print("Not have a cat")
    print("Similarity Rate:", srate)
