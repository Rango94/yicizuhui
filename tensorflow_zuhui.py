import tensorflow as tf
import random as rd
import numpy as np

X=np.load('X_3_huan.npy')
print(X.shape)
Y=np.load('Y_huan.npy')
ss=int(len(X)*0.2)
X_test=X[:ss]
X=X[ss:]

Y_tmp=[]
for idx,i in enumerate(Y):
    tmp=np.zeros(2)
    tmp[i]=1
    Y_tmp.append(tmp)
Y=np.array(Y_tmp)
Y_test=Y[:ss]
Y=Y[ss:]
#w1=tf.Variable(tf.random_uniform([3,2],minval=-0.5,maxval=0.5))
w1=tf.constant([[1,0],[0,1],[0,0]],dtype=tf.float32)
w_y1=tf.Variable(tf.random_uniform([2,4],minval=-0.5,maxval=0.5))
w_y2=tf.Variable(tf.random_uniform([4,2],minval=-0.5,maxval=0.5))
w2=tf.Variable(tf.random_uniform([2,2],minval=-0.5,maxval=0.5))

x=tf.placeholder(tf.float32,shape=(None,3))
y_=tf.placeholder(tf.float32,shape=(None,2))

a=tf.matmul(x,w1)

b=tf.nn.relu(tf.matmul(a,w_y1))
c=tf.nn.relu(tf.matmul(b,w_y2))
y=tf.matmul(c,w2)

loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y)#+tf.contrib.layers.l2_regularizer(0.001)(w1)

train_step=tf.train.AdadeltaOptimizer(1).minimize(loss)

batch_size=500
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    STEP=100000
    for i in range(STEP):
        start=rd.randint(0,len(X)-1)
        end=min(len(X),start+batch_size)
        # start = 0
        # end = len(X)
        sess.run(train_step,feed_dict={x:X,y_:Y})
        if STEP%1000==0:
            y___=sess.run(y, feed_dict={x:X_test, y_:Y_test})
            y___=y___.tolist()
            r=0
            Y_test_=Y_test.tolist()
            for idx,i in enumerate(y___):
               if  Y_test_[idx].index(max(Y_test_[idx]))==y___[idx].index(max(y___[idx])):
                   r+=1
            print(np.mean(sess.run(loss, feed_dict={x:X_test, y_:Y_test})),r/len(Y_test))
            np.save('w_huan_nonlinear_best.npy',sess.run(w1))
        #0.15456033 0.9425



