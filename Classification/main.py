# -*- coding: utf-8 -*-
"""
Created on Thu Dec 01 11:41:27 2016

@author: sindhura
"""

import cPickle
import gzip
import math
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


files = 'mnist.pkl.gz'
fo = gzip.open(files, 'rb')
training_data, valid_data, test_data = cPickle.load(fo)
#print("Training Data: ",training_data[0])

x_vector = training_data[0]
t_vector = training_data[1]


xvalid_vector = valid_data[0]
tvalid_vector = valid_data[1]

xtest_vector = test_data[0]
ttest_vector = test_data[1]

a1 = x_vector.shape[0]
a2 = x_vector.shape[1]
a3 = xvalid_vector.shape[0]
a4 = xvalid_vector.shape[1]
a5 = xtest_vector.shape[0]
a6 = xtest_vector.shape[1]

b1 = t_vector.shape[0]
#b2 = t_vector.shape[1]
b3 = tvalid_vector.shape[0]
#b4 = tvalid_vector.shape[1]
b5 = ttest_vector.shape[0]
#b6 = ttest_vector.shape[1]

#print("length of x: ",x_vector.shape)
#print("length of t: ",t_vector.shape)

tj_vector = np.zeros((b1,10),dtype='float')


for i in range(0,a1):
    t_val = t_vector[i]
#    print("t_val: ",t_val)
    tj_vector[i][t_val]=1
#    print("tj_vector: ",tj_vector[i])
    
def Proj3(x_vector,t_vector,tj_vector,xvalid_vector,tvalid_vector,xtest_vector,ttest_vector,a1,a2,a3,a4,a5,a6,b1,b3,b5):

    weights,y_vector = Logistic_Regression(t_vector,x_vector,tj_vector,a1,a2) 
    Eval = LR_Eval(y_vector,a1,t_vector)
    print("Evaluation of Logistic Regression: ",Eval)    
        
    wji_vector,wkj_vector,y1_vec = SNN(t_vector,x_vector,tj_vector,a1,100,a2)
    Eval1 = SNN_Eval(y1_vec,a1,t_vector)
    print("Evaluation of Single Layer Neural Network: ",Eval1)
    print("Shape of wji: ",wji_vector.shape)    
    
    #print("a3: ",a3)
    #print("a4: ",a4)
    print("Weights of Logistic Regression: ",weights)
    print("Weights wji in Single Layer Neural Network: ",wji_vector)
    print("Weights wkj in Single Layer Neural Network: ",wkj_vector) 
   
    Eval2 = LR_VT(xvalid_vector,weights,tvalid_vector,a3)
    print("Evaluation of Logistic Regression Validation: ",Eval2)    
    #Evalvalid_LR = LR_Eval(y_vector,a3,)
    
    Eval3 = SNN_VT(xvalid_vector,wji_vector,wkj_vector,tvalid_vector,a3,100,a4)
    print("Evaluation of Single Layer Neural Network Validation: ",Eval3)
    
    Eval4 = LR_VT(xtest_vector,weights,ttest_vector,a5)
    print("Evaluation of Logistic Regression Test: ",Eval4)
    
    Eval5 = SNN_VT(xtest_vector,wji_vector,wkj_vector,ttest_vector,a5,100,a6)
    print("Evaluation of Single Layer Neural Network Test: ",Eval5)


def Logistic_Regression(t_vector,x_vector,tj_vector,N,M):    
    
    weights = np.random.rand(M,10)
    
   # print("Weights: ",weights)
   # print("Weights vector length: ",weights.shape)
    
    b_vector = np.ones((N,10),dtype='float')
    y_vector = np.zeros((N,10),dtype='float')
    ysub_vector = np.zeros((N,10),dtype='float')
    ysub_transpose = np.zeros((10,N),dtype='float')

    
    for i in range(0,500):
        x1 = x_vector.dot(weights)
      #  print("XW: ",x1)
        a_vector = x1 + b_vector
      #  print("Activation vector: ",a_vector)
      #  print("Length of activation vector: ",a_vector.shape)
        
        for j in range(0,N):
            a_sum = 0
            for k in range(0,10):
                a_sum = a_sum +  (math.exp(a_vector[j][k]))
            for k1 in range(0,10):
                y_vector[j][k1] = ((math.exp(a_vector[j][k1])) / a_sum)
        
      #  print("Y: ",y_vector)
      #  print("Length of Y: ",y_vector.shape)
        
        x_transpose = np.transpose(x_vector)
        mean = np.mean(x_transpose,axis=0)
      #  print(mean)
      #  print("Length of Transpose of X: ",x_transpose.shape)
        ysub_vector = y_vector - tj_vector
      #  print("Y-T: ",ysub_vector)
       # ysub_transpose = np.transpose(ysub_transpose)
        ErrGrad = x_transpose.dot(ysub_vector)
      #  ErrGrad = ysub_vector.dot(x_vector)
      #  print("Gradient of the error function: ",ErrGrad)
      #  print("Length of Error gradient: ",ErrGrad.shape)
            
        eta = 0.5
        e1 = (eta * ErrGrad) / N
        weights_new = weights - e1
        
      #  print("e1: ",e1)
        weights = weights_new
        
        
    return weights,y_vector
    #    print("Prev weights: ",weights)
    
    #print("New weights: ",weights_new)
       
       
    #print("First row of y vector: ",y_vector[0])
    #n1 = np.argmax(y_vector[0])
    #print("Index by argmax: ",n1)
       
def LR_Eval(y_vector,N,t_vector):
    
    y_final = np.zeros((N,1),dtype='float')
    
    for i in range(0,N):
        
        n = np.argmax(y_vector[i])
        y_final[i] = n
    
    #print("Final y: ",y_final)
    
    count = 0
    for i in range(0,N):
     #   print("i:",i)
     #   print("y_final[i]",y_final[i])
     #   print("t_vector[i]",t_vector[i])
        if y_final[i] == t_vector[i]:
            count = count+1
    
    N_right = count
    print("N_right: ",N_right)
    Eval = float(N_right) / float(N)
    print("Eval: ",float(Eval)*100)
    
    return Eval


#...............Single Layer Neural Network...................

def SNN(t_vector,x_vector,tj_vector,N,M,O):

    
    
    wji_vector = np.random.rand(O,M)*0.1
    b_wji = np.ones((1,M),dtype='float')
    b_wkj = np.ones((1,10),dtype='float')
    zj_vector = np.zeros((1,M),dtype='float')
    wkj_vector = np.random.rand(M,10)*0.1
    y1_vector = np.zeros((1,10),dtype='float')
    y1_vec = np.zeros((N,10),dtype='float')
    x_transpose1 = np.zeros((O,1),dtype='float')
    
    for k in range(5):
        print("Iteration: ",k)
        for i in range(0,N):
            for j in range(0,O):
                x_transpose1[j][0] = x_vector[i][j]
                
           # x_transpose1 = np.transpose(x_vector[i])
            #print("Tranpose of x shape: ",x_transpose1.shape)
            xw_vector = x_vector[i].dot(wji_vector)
          #  print("xw_vector: ",xw_vector)
            zj_vector = (xw_vector + b_wji)
           # print("Length of zj: ",zj_vector.shape)
         #   print("zj: ",zj_vector)
            zj_vector = 1/(1+np.exp(-zj_vector))
         #   print("zj_vector after big loop: ",zj_vector)
            
            
            wkj_zj = zj_vector.dot(wkj_vector)
            ak_vector = (wkj_zj + b_wkj)
            
            # print("ak_vector: ",ak_vector)
          #  print("Length of ak_vector: ",ak_vector.shape)
        
            ak_sum = 0
            for k in range(0,10):
                ak_sum = ak_sum +  (math.exp(ak_vector[0][k]))
            #     print("ak_sum: ",ak_sum)
            for k1 in range(0,10):
                y1_vector[0][k1] = ((math.exp(ak_vector[0][k1])) / ak_sum)
            
            for j in range(0,10):
                y1_vec[i][j] = y1_vector[0][j]
                    
          #  print("y1_vector: ",y1_vector)
                
            #..................Derivation w.r.t wji.....................
         #   print("Shape of tj_vector: ",tj_vector[i].shape)
         #   print("y1 Shape: ",y1_vector.shape)
            diff1 = y1_vector - tj_vector[i]
         #   print("Diff shape: ",diff1.shape)
            hzj_vector = zj_vector
            hzj1_vector = 1 - hzj_vector
            hz_transpose = np.transpose(hzj1_vector)
            hder = hzj_vector.dot(hz_transpose)
            wkj_transpose = np.transpose(wkj_vector)
            mul1 = x_transpose1.dot(diff1)
            mul2 = mul1.dot(wkj_transpose)
            wji1_vector = hder * mul2
            eta = 0.01
            sgd1 = (eta * wji1_vector)
        
            wji_vector = wji_vector - sgd1
            
            #..................Derivation w.r.t wkj.....................fd
            
            zj_transpose = np.transpose(zj_vector)
            wkj1_vector = zj_transpose.dot(diff1)
            sgd2 = (eta * wkj1_vector) 
            
            wkj_vector = wkj_vector - sgd2
    
    return wji_vector,wkj_vector,y1_vec
            
    #print("y1_vector: ",y1_vector)
            
def SNN_Eval(y1_vec,N,t_vector):
    
    y_final1 = np.zeros((N,1),dtype='float')
    for i in range(0,N):
        n = np.argmax(y1_vec[i])
        y_final1[i] = n
    
   # print("Final y: ",y_final1)
    
    count1 = 0
    for i in range(0,N):
     #   print("i:",i)
     #   print("y_final[i]",y_final[i])
     #   print("t_vector[i]",t_vector[i])
        if y_final1[i] == t_vector[i]:
            count1 = count1+1
    
    N_right1 = count1
    print("N_right: ",N_right1)
    Eval1 = float(N_right1) / float(N)
    print("Eval: ",float(Eval1)*100)
    
    return Eval1

def LR_VT(x_vector,weights,t_vector,N):
    
    b_vector = np.ones((N,10),dtype='float')
    y_vector = np.zeros((N,10),dtype='float')
    ysub_vector = np.zeros((N,10),dtype='float')
    ysub_transpose = np.zeros((10,N),dtype='float')

    tj_vector = np.zeros((b1,10),dtype='float')


    for i in range(0,N):
        t_val = t_vector[i]
        #    print("t_val: ",t_val)
        tj_vector[i][t_val]=1    

    x1 = x_vector.dot(weights)
    #  print("XW: ",x1)
    a_vector = x1 + b_vector
    for j in range(0,N):
        a_sum = 0
        for k in range(0,10):
            a_sum = a_sum +  (math.exp(a_vector[j][k]))
        for k1 in range(0,10):
            y_vector[j][k1] = ((math.exp(a_vector[j][k1])) / a_sum) 
    
    y_final = np.zeros((N,1),dtype='float')
    
    for i in range(0,N):
        
        n = np.argmax(y_vector[i])
        y_final[i] = n
    
    #print("Final y: ",y_final)
    
    count = 0
    for i in range(0,N):
     #   print("i:",i)
     #   print("y_final[i]",y_final[i])
     #   print("t_vector[i]",t_vector[i])
        if y_final[i] == t_vector[i]:
            count = count+1
    
    N_right = count
    print("N_right: ",N_right)
    Eval = float(N_right) / float(N)
    print("Eval: ",float(Eval)*100)
    
    return Eval
    
def SNN_VT(x_vector,wji_vector,wkj_vector,t_vector,N,M,O):
    
    
    tj_vector = np.zeros((b1,10),dtype='float')


    for i in range(0,N):
        t_val = t_vector[i]
        #    print("t_val: ",t_val)
        tj_vector[i][t_val]=1
        
        
  #  wji_vector = np.random.rand(O,M)*0.1
    b_wji = np.ones((1,M),dtype='float')
    b_wkj = np.ones((1,10),dtype='float')
    zj_vector = np.zeros((1,M),dtype='float')
   # wkj_vector = np.random.rand(M,10)*0.1
    y1_vector = np.zeros((1,10),dtype='float')
    y1_vec = np.zeros((N,10),dtype='float')
    x_transpose1 = np.zeros((O,1),dtype='float')
    
    for i in range(0,N):
        for j in range(0,O):
            x_transpose1[j][0] = x_vector[i][j]
                
           # x_transpose1 = np.transpose(x_vector[i])
            #print("Tranpose of x shape: ",x_transpose1.shape)
        xw_vector = x_vector[i].dot(wji_vector)
       # print("xw_vector shape: ",xw_vector.shape)
          #  print("xw_vector: ",xw_vector)
        zj_vector = (xw_vector + b_wji)
           # print("Length of zj: ",zj_vector.shape)
         #   print("zj: ",zj_vector)
        zj_vector = 1/(1+np.exp(-zj_vector))
         #   print("zj_vector after big loop: ",zj_vector)
            
            
        wkj_zj = zj_vector.dot(wkj_vector)
        ak_vector = (wkj_zj + b_wkj)
            
            # print("ak_vector: ",ak_vector)
          #  print("Length of ak_vector: ",ak_vector.shape)
        
        ak_sum = 0
        for k in range(0,10):
            ak_sum = ak_sum +  (math.exp(ak_vector[0][k]))
            #     print("ak_sum: ",ak_sum)
        for k1 in range(0,10):
            y1_vector[0][k1] = ((math.exp(ak_vector[0][k1])) / ak_sum)
            
        for j in range(0,10):
            y1_vec[i][j] = y1_vector[0][j]    
    
    y_final1 = np.zeros((N,1),dtype='float')
    for i in range(0,N):
        n = np.argmax(y1_vec[i])
        y_final1[i] = n
    
    #print("Final y: ",y_final1)
    
    count1 = 0
    for i in range(0,N):
     #   print("i:",i)
     #   print("y_final[i]",y_final[i])
     #   print("t_vector[i]",t_vector[i])
        if y_final1[i] == t_vector[i]:
            count1 = count1+1
    
    N_right1 = count1
    print("N_right: ",N_right1)
    Eval1 = float(N_right1) / float(N)
    print("Eval: ",float(Eval1)*100)
    
    return Eval1


#............Calling Proj3 to implement logistic regression and SNN...........
    
Proj3(x_vector,t_vector,tj_vector,xvalid_vector,tvalid_vector,xtest_vector,ttest_vector,a1,a2,a3,a4,a5,a6,b1,b3,b5)
    

#...................Convolutional Neural Networks......................


mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

x = tf.placeholder("float",shape=[None,784])
y_ = tf.placeholder("float",shape=[None,10])
    
def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = 'SAME')
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides=[1,2,2,1],padding='SAME')
    
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

x_image = tf.reshape(x,[-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

#Fully connected layer
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)
W_fc2 = weight_varaible([1024,10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)


cross_entropy = -tf.reduce.sum(y_*tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))

init = tf.initialize_all_variables()
sess = tf.InteractiveSession()
sess.run(init)
for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i%100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1], keep_prob: 1.0})
        print "step %d, training accuracy %g"%(i,train_accuracy)
    train_step.run(feed_dict={x: batch[0],y_: batch[1], keep_prob: 0.5})
    
print "test accuracy %g"%accuracy.eval(feed_dict={ a: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
    
    



fo.close()