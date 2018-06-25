import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

def genarete(style='ball'):
    X=[]
    Y=[]
    if style=='ball':
        for i in range(100000):
            if i%5000==0:
                print(i)
            tmp=(np.random.random(3)-0.5)*6
            x,y,z=tmp[0],tmp[1],tmp[2]
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
        X=np.array(X)
        Y=np.array(Y)
        np.save('X_3' + '_' + style, X)
        np.save('Y' + '_' + style, Y)
    if style=='cubu':
        X = []
        Y = []
        for i in range(100000):
            if i % 5000 == 0:
                print(i)
            tmp = (np.random.random(3) - 0.5) * 6
            x, y, z = tmp[0], tmp[1], tmp[2]
            if x>0.75 and y>0.75 and z>0.75:
                X.append(tmp)
                Y.append(0)
            if x<-0.75 and y>0.75 and z>0.75:
                X.append(tmp)
                Y.append(1)
            if x>0.75 and y<-0.75 and z>0.75:
                X.append(tmp)
                Y.append(2)
            if x<-0.75 and y<-0.75 and z>0.75:
                X.append(tmp)
                Y.append(3)
            if x>0.75 and y>0.75 and z<-0.75:
                X.append(tmp)
                Y.append(4)
            if x<-0.75 and y>0.75and z<-0.75:
                X.append(tmp)
                Y.append(5)
            if x>0.75 and y<-0.75 and z<-0.75:
                X.append(tmp)
                Y.append(6)
            if x<-0.75 and y<-0.75 and z<-0.75:
                X.append(tmp)
                Y.append(7)
        X = np.array(X)
        Y = np.array(Y)
        np.save('X_3'+'_'+style, X)
        np.save('Y'+'_'+style,Y)
    if style=='huan':
        X = []
        Y = []
        for i in range(1000000):
            if i % 5000 == 0:
                print(i)
            tmp = (np.random.random(3) - 0.5) * 8
            x, y, z = tmp[0], tmp[1], tmp[2]
            if x*x+y*y<4 and x*x+y*y>2 and z>0.5 and z<2:
                X.append(tmp)
                Y.append(0)
            if x*x+y*y+z*z<1:
                X.append(tmp)
                Y.append(1)
        X = np.array(X)
        Y = np.array(Y)
        np.save('X_3'+'_'+style, X)
        np.save('Y'+'_'+style,Y)

def genareta_random_w():
    w = np.random.random((3, 2))
    np.save('w_random', w)



def plot3(X_name,Y_name):
    X = np.load(X_name + '.npy')
    Y = np.load(Y_name + '.npy')
    ax = plt.subplot(111,projection='3d')  # 创建一个三维的绘图工程
    dic={0:'red',1:'green',2:'blue',3:'yellow',4:'black',5:'pink',6:'gray',7:'purple'}
    for k in range(np.max(Y)+1):
        X_tmp=[]
        for idx,i in enumerate(X):
            if Y[idx]==k:
                X_tmp.append(i)
        X_tmp=np.array(X_tmp)
        ax.scatter(X_tmp[:,0], X_tmp[:,1],X_tmp[:,2] ,s=10,c=dic[k])  # 绘制数据点
    ax.set_zlabel('z')
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    ax.view_init(elev=90, azim=90)
    plt.show()


def plot2(X_name,Y_name,w_name):
    w = np.load(w_name+'.npy')
    print(w)
    X = np.load(X_name+'.npy')
    Y=np.load(Y_name+'.npy')
    X = np.dot(X, w)
    ax = plt.subplot(111)  # 创建一个三维的绘图工程
    dic = {0: 'red', 1: 'green', 2: 'blue', 3: 'yellow', 4: 'black', 5: 'pink', 6: 'gray', 7: 'purple'}
    for k in range(np.max(Y)+1):
        X_ = []
        for idx, i in enumerate(X):
            if Y[idx] == k:
                X_.append(i)
        X_ = np.array(X_)
        ax.scatter(X_[:, 0], X_[:, 1], s=10, c=dic[k])  # 绘制数据点
    ax.set_ylabel('Y')
    ax.set_xlabel('X')
    plt.show()



# genarete('huan')
#genareta_random_w()
# plot3('X_3_huan','Y_huan')
plot2('X_3_huan','Y_huan','w_huan_nonlinear_best')
