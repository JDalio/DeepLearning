import numpy as np
import pylab as plt
from init_utils import *

train_X, train_Y, test_X, test_Y = load_dataset()
# (2,300)    (1,300)    (2,100)    (1,100)
# train_X为二维坐标上的点, train_Y表明点的两种颜色

'''
plt窗口，figure窗口上的图，图上有坐标轴
close关窗口，clf清图，cla清坐标轴
'''
# plt.figure(1)
# plt.axis([xmin,xmax,ymin,ymax])
plt.title("Classify Two Kinds of Nodes")
for i in range(train_X.shape[1]):
    if train_Y[0][i] == 1:
        plt.plot(train_X[0][i], train_X[1][i], 'ob')
    else:
        plt.plot(train_X[0][i], train_X[1][i], 'or')
# plt.savefig('a.png')
plt.show()
plt.close('all')
