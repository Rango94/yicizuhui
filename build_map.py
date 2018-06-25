import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

X=[]
Y=[]
for i in range(100000):
    if i%5000==0:
        print(i)
    tmp=(np.random.random(3)-0.5)*6
    x,y,z=tmp[0],tmp[1],tmp[2]
    # if x>0.75 and y>0.75 and z>0.75:
    #     X.append(tmp)
    #     Y.append(0)
    # if x<-0.75 and y>0.75 and z>0.75:
    #     X.append(tmp)
    #     Y.append(1)
    # if x>0.75 and y<-0.75 and z>0.75:
    #     X.append(tmp)
    #     Y.append(2)
    # if x<-0.75 and y<-0.75 and z>0.75:
    #     X.append(tmp)
    #     Y.append(3)
    # if x>0.75 and y>0.75 and z<-0.75:
    #     X.append(tmp)
    #     Y.append(4)
    # if x<-0.75 and y>0.75and z<-0.75:
    #     X.append(tmp)
    #     Y.append(5)
    # if x>0.75 and y<-0.75 and z<-0.75:
    #     X.append(tmp)
    #     Y.append(6)
    # if x<-0.75 and y<-0.75 and z<-0.75:
    #     X.append(tmp)
    #     Y.append(7)
    if math.pow((x+2),2)+math.pow((y+2),2)+math.pow((z+2),2)<1:
        X.append(tmp)
        Y.append(0)
    if math.pow((x+2),2)+math.pow((y-2),2)+math.pow((z+2),2)<1:
        X.append(tmp)
        Y.append(1)
    if math.pow((x-2),2)+math.pow((y+2),2)+math.pow((z+2),2)<1:
        X.append(tmp)
        Y.append(2)
    if math.pow((x-2),2)+math.pow((y-2),2)+math.pow((z+2),2)<1:
        X.append(tmp)
        Y.append(3)
    if math.pow((x+2),2)+math.pow((y+2),2)+math.pow((z-2),2)<1:
        X.append(tmp)
        Y.append(4)
    if math.pow((x+2),2)+math.pow((y-2),2)+math.pow((z-2),2)<1:
        X.append(tmp)
        Y.append(5)
    if math.pow((x-2),2)+math.pow((y+2),2)+math.pow((z-2),2)<1:
        X.append(tmp)
        Y.append(6)
    if math.pow((x-2),2)+math.pow((y-2),2)+math.pow((z-2),2)<1:
        X.append(tmp)
        Y.append(7)






def plot2():
    w = np.random.random((3, 2)) * 2 - 1
    np.save('w', w)
    X = np.load('X_3.npy')
    Y = np.load('Y.npy')
    X=np.dot(X,w)
    ax = plt.subplot(111)  # 创建一个三维的绘图工程
    dic = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'black', 5: 'pink', 6: 'gray', 7: 'purple'}
    for k in range(8):
        X_ = []
        for idx, i in enumerate(X):
            if Y[idx] == k:
                X_.append(i)
        X_ = np.array(X_)
        ax.scatter(X_[:, 0], X_[:, 1], s=10, c=dic[k])  # 绘制数据点

    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    #ax.view_init(elev=5, azim=10)
    plt.show()


def plot3(X,Y):
    np.save('X_3',X)
    X=np.array(X)
    np.save('X_2',X)
    np.save('Y',Y)
    ax = plt.subplot(111,projection='3d')  # 创建一个三维的绘图工程
    dic={0:'red',1:'green',2:'blue',3:'yellow',4:'black',5:'pink',6:'gray',7:'purple'}
    for k in range(8):
        X_=[]
        for idx,i in enumerate(X):
            if Y[idx]==k:
                X_.append(i)
        X_=np.array(X_)
        ax.scatter(X_[:,0], X_[:,1],X_[:,2] ,s=10,c=dic[k])  # 绘制数据点

    ax.set_zlabel('z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.view_init(elev=135, azim=30)
    plt.show()

def plot2_best():
    w = np.load('w_best.npy')
    X = np.load('X_3.npy')
    Y=np.load('Y.npy')
    X = np.dot(X, w)
    ax = plt.subplot(111)  # 创建一个三维的绘图工程
    dic = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'black', 5: 'pink', 6: 'gray', 7: 'purple'}
    for k in range(8):
        X_ = []
        for idx, i in enumerate(X):
            if Y[idx] == k:
                X_.append(i)
        X_ = np.array(X_)
        ax.scatter(X_[:, 0], X_[:, 1], s=10, c=dic[k])  # 绘制数据点
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()

#plot2()
#plot3(X,Y)
plot2_best()
