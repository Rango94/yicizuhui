import tensorflow as tf
import random as rd
import numpy as np

X=np.dot(np.load('X_3.npy'),np.load('w.npy'))

Y=np.load('Y.npy')
X_test=X[:2000]

X=X[2000:]

Y_tmp=[]
for idx,i in enumerate(Y):
    tmp=np.zeros(8)
    tmp[i]=1
    Y_tmp.append(tmp)
Y=np.array(Y_tmp)
Y_test=Y[:2000]
Y=Y[2000:]
w1=tf.Variable(tf.random_uniform([2,8],minval=-0.5,maxval=0.5))

x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,8))

y=tf.matmul(x,w1)

loss=tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_,logits=y)+tf.contrib.layers.l2_regularizer(0.001)(w1)

train_step=tf.train.AdadeltaOptimizer(1).minimize(loss)

batch_size=100
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    STEP=100000
    for i in range(STEP):
        start=rd.randint(0,len(X)-1)
        end=min(len(X),start+batch_size)
        sess.run(train_step,feed_dict={x:X,y_:Y})
        if STEP%10000==0:
            y___=sess.run(y, feed_dict={x:X_test, y_:Y_test})

            y___=y___.tolist()
            r=0
            Y_test_=Y_test.tolist()
            for idx,i in enumerate(y___):
               if  Y_test_[idx].index(max(Y_test_[idx]))==y___[idx].index(max(y___[idx])):
                   r+=1
            print(np.mean(sess.run(loss, feed_dict={x:X_test, y_:Y_test})),r/len(Y_test))
            #0.66701466 0.741




