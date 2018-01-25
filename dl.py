import cv2
import os
import numpy as np
import tensorflow as tf

def CompressImg(dir):
    matrix = np.zeros([200 * 200, 1],dtype='f8')
    files = os.listdir(dir)
    for f in files:
        path=dir + "/"+f
        if not os.path.isdir(path):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img=np.array(img,dtype='f8')
            img=img.reshape(img.size,1)
            matrix=np.hstack([matrix,img])
        else:
            matrix=np.hstack([matrix,CompressImg(path)])
    return matrix[...,1:]

def relu(A):
    return np.maximum(A,np.zeros(A.shape))
#set=CompressImg('dataset')
img=cv2.imread('dataset.png',cv2.IMREAD_GRAYSCALE)
set=np.array(img,dtype='f8')

T1=np.hstack([np.ones([1,30]),np.zeros([1,10])])
T2=np.zeros([1,40])
T2[0][6:10]=1
T2[0][39]=1
T2[0][35]=1
T=np.vstack([T1,T2])
len=set.shape[0]
features=2
W=np.random.randn(len,features)
B=np.random.randn(features,40)
for i in range(10):
    #Forward Propegation
    Y=np.dot(W.T,set)+B
    a=relu(Y)
    print(a)
'''
    loss=-(T*np.log(a)+(1-T)*np.log(1-a))
    cost=np.mean(loss,1).reshape(2,1)
    #backward Propegation
    rate=0.5
    db=(a-T)
    dw=np.dot(set,(a-T).T)
    W-=rate*dw
    B-=rate*db
    print(loss)
'''
