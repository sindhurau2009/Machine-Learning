# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 01:47:40 2016

@author: sindhura
"""

import numpy as np
import re
import random
#import matplotlib.pyplot as plt

M=4



TrainN=55601
ValidN=62612
valid = ValidN - TrainN
TestN=69623
Test = TestN - ValidN
data = np.genfromtxt("Querylevelnorm.txt",dtype=None,delimiter=None)
Y=np.zeros((TrainN,1),dtype='f8')
X=np.zeros((TrainN,46),dtype='float')
Yvalid=np.zeros((valid,1),dtype='f8')
Xvalid=np.zeros((valid,46),dtype='float')
#print("XValid size: ",Xvalid.shape)
#print("YValid size: ",Yvalid.shape)

Ytest=np.zeros((Test,1),dtype='f8')
Xtest=np.zeros((Test,46),dtype='float')

X2 = np.zeros((TrainN,46),dtype='float')
Y2 = np.zeros((TrainN,1),dtype='f8')
X2Valid = np.zeros((valid,46),dtype='float')
Y2Valid = np.zeros((valid,1),dtype='f8')
X2Test = np.zeros((Test,46),dtype='float')
Y2Test = np.zeros((Test,1),dtype='f8')

W1=np.zeros((M,1),dtype='float')
WR1=np.zeros((M,1),dtype='float')

#Weight = np.zeros((1,4),dtype='float')

print("UBIT Name: suppu")
print("Person Number: 50206730")
lamda=random.uniform(0,1)

#------------------Obtaining X and Y matrix---------------------
for row in range(0,TrainN):
    j=0
    for k in range(0,48):
        if k==0:
            Y[row]=data[row][k]
        else:
            if k!=1:
                t=data[row][k]
                t=re.split('[:]',t)
                X[row][j]=t[1]
                j=j+1 

y=0
for row in range(TrainN,ValidN):
    j=0
    x=0
    
    for k in range(0,48):
        
        if k==0:
            Yvalid[x]=data[row][k]
            x = x+1
        else:
            if k!=1:
                t=data[row][k]
                t=re.split('[:]',t)
                Xvalid[y][j]=t[1]
                
                j=j+1 
    y = y+1
          
y1=0
for row in range(ValidN,TestN):
    j=0
    x1=0
    for k in range(0,48):
        if k==0:
            Ytest[x1]=data[row][k]
            x1=x1+1
        else:
            if k!=1:
                t=data[row][k]
                t=re.split('[:]',t)
                Xtest[y1][j]=t[1]
                j=j+1
    y1=y1+1
    
"""
def plotting():
    
    Mv = [[3],[4],[5],[6],[7],[9]]
    ERMSv = [[0.349],[0.363],[0.233],[0.350],[0.465],[0.293]]
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(Mv,ERMSv)
    plt.savefig('images/MvERMSv.jpeg')
    return
"""
    
#plotting()

def LeToR1(X,Y,XV,YV,XT,YT,N1,N2,N3):
    
    phi1 = phi(X,Y,N1)
    W1 = weight_closed(X,Y,phi1)
    ERMS = Ermscal(W1,X,Y,N1,phi1) 
    print("Weight of Closed form sol: ",W1)
    
       
    WR1 = regweight_closed(X,Y,phi1)
    RERMS = Ermscal(WR1,X,Y,N1,phi1)
    print("Weight of Reg Closed Form Sol: ",WR1)
    
    
    phi2=phi(XV,YV,N2)
    ERMSV = Ermscal(W1,XV,YV,N2,phi2)

    
    RERMSV = Ermscal(WR1,XV,YV,N2,phi2)

    
    phi3 = phi(XT,YT,N3)
    ERMST = Ermscal(W1,XT,YT,N3,phi3)

    
    RERMST = Ermscal(WR1,XT,YT,N3,phi3)

    
    WS = Stochastic(X,Y,phi1,N1,W1)
    SERMS = Ermscal(WS,X,Y,N1,phi1)
    print("Weight of Stochastic: ",WS)

    
    WSR = Reg_Stochastic(X,Y,N1,phi1)
    SRERMS = Ermscal(WSR,X,Y,N1,phi1)
    print("Weight of Stochastic Reg: ",WSR)

    
    phi4=phi(XV,YV,N2)
    SERMSV = Ermscal(WS,XV,YV,N2,phi4)

    
    SRERMSV = Ermscal(WSR,XV,YV,N2,phi4)

    
    phi5 = phi(XT,YT,N3)
    SERMST = Ermscal(WS,XT,YT,N3,phi5)

    
    SRERMST = Ermscal(WSR,XT,YT,N3,phi5)

    
    print("ERMS Closed form training set: ",ERMS)
    print("Regularized ERMS Closed form training set: ",RERMS)
    print("ERMS Closed form validation set: ",ERMSV)
    print("Regularized ERMS Closed form validation set: ",RERMSV)
    print("ERMS Closed form testing set: ",ERMST)
    print("Regularized ERMS Closed form testing set: ",RERMST)
    print("ERMS Stochastic form training set: ",SERMS)
    print("Regularized ERMS Stochastic from training set: ",SRERMS)
    print("ERMS Stochastic form validation set: ",SERMSV)
    print("Regularized ERMS Stochastic form validation set: ",SRERMSV)    
    print("ERMS Stochastic form testing set: ",SERMST)
    print("Regularized ERMS Stochastic form testing set: ",SRERMST)    
    
    return

def phi(X,Y,N):
    
    sigma=np.zeros((46,46),dtype='float')
    for i in range(0,46):
        sigma[i][i] = np.var(X[:,i])
        #print(sigma)
        #
        #M=int(M)
    #print("Sigma: ",sigma)
       
    mu=np.zeros((M,46))
    #mu=1/46
    """
    for k in range(0,TrainN):
        for n in range(0,4):
            Z[k,n]=X[k,n]
            """
    #----------------------Generating Means----------------------
    #print("Mean:")
    for k in range(0,M):
        n=random.randint(0,N)
        mu[k]=X[n]
        #print(mu[k])
    #print("mu",len(mu[:,1]),len(mu[1,:]))
    #-----------------Calculating phi matrix----------------------
    phi=np.zeros((N,M),dtype='float')
    for i in range(0,N):
        for j in range(0,M):
            if(j==0):
                phi[i,j]=1
            else:
                t=(X[i]-mu[j])
                #print("t",t)
                t2=np.transpose(t)
                #print("t2",t2)
                t3=sigma.dot(t)
                #print("t3",t3)
                t4=t2.dot(t3)
                #print("t4",t4)
                #t4=t*t3
                t5=float(t4/2)
                phi[i,j]=(np.exp(-t5))
  #  print("phi: ",phi)
                
    return phi
    
def weight_closed(X,Y,phi):
    
    #------------------Calucalting Weights--------------
    W=np.zeros((M,1),dtype='float')
    p1=np.transpose(phi)
    p2=p1.dot(phi)
    p3=np.linalg.inv(p2)
    p4=p3.dot(p1)
    W=p4.dot(Y)
   # print("W",W)
    
    return W
    
def regweight_closed(X,Y,phi):
    
    #--------------------Calculating Regularized weights-------
    L=np.zeros((M,M),dtype="int")
    for i in range(0,M):
        L[i,i]=1
    p1=np.transpose(phi)
    p2=p1.dot(phi)
    L=lamda*L
    p3=L+p2
    p3=np.linalg.inv(p3)
    p4=p3.dot(p1)
    WR=p4.dot(Y)
   # print("WR",WR)
    
    return WR
    
def Ermscal(W,X,Y,N,phi):
    
    #--------------------Calculating EW--------------------
    #print("W:",W)
    w1=np.transpose(W)
    #print("W1 : ",w1)
  #  w2=w1.dot(W)
  #  EW=float(w2/2)
    #print("EW",EW)
    #--------------------Calculating ED--------------------
    h2=0
    #print("w1,phi[1]",w1,phi[1])
    for i in range(0,N):
        h=w1.dot(phi[i])
        #print("h: ",h)
        h2+=(Y[i]-h)*(Y[i]-h)  #----------EDW is being calculated--------
        #print("h2",h2)
    EDW=float(h2/2)
    #print("EDW",EDW)
    #--------------------Calculating ERMS-------------------
    ER1=(2*EDW)/N
    ERMS=np.sqrt(ER1)
  #  print("ERMS",ERMS)
    
    return ERMS
    
def Stochastic(X,Y,phi,N,W):
    
    W1 = np.zeros((M,1),dtype="float")
    W1[1] = [1]
    Erms1 = Ermscal(W1,X,Y,N,phi)
   # print("ERMS1: ",Erms1)
    ph1=np.zeros((M,1),dtype="float")
   # print("W1",W1)
    eta = 0.5
    for x in range(0,100):
     #   eta = 0.5 * eta
        ph = phi[x]
      #  print("ph",ph)
        for i in range(0,M):
            ph1[i]=ph[i]
        #print("Transpose",np.transpose(phi[x]))
        #ph1 = np.transpose(phi[x])
       # print("ph1",ph1)
        W2 = np.transpose(W1)
      #  print("pW2",W2)
        t2 = Y[x] - (W2.dot(ph1))
     #   print("t2",t2)
        t3 = - (ph1.dot(t2))
    #    print("t3: ",t3)
        
        dw = (eta*t3)
        de = dw / N
        W2 = W1 - de
    #    print("W2: ",W2)
        Erms2 = Ermscal(W2,X,Y,N,phi)
    #    print("ERMS2: ",Erms2)
       # if Erms2 - Erms1 <= 0.00001:
        #    break;
        Wt = W1
        W1 = W2
        
        
    return W2
    
    
def Reg_Stochastic(X,Y,N,phi):
    
    W1 = np.zeros((M,1),dtype="float")
    W1[1] = [1]
    ew = np.zeros((M,1),dtype="float")
    Erms1 = Ermscal(W1,X,Y,N,phi)
    #print("ERMS1: ",Erms1)
    ph1=np.zeros((M,1),dtype="float")
    #print("W1",W1)
    eta = 0.5
    for x in range(0,100):
     #   eta = 0.5 * eta
        ph = phi[x]
      #  print("ph",ph)
        for i in range(0,M):
            ph1[i]=ph[i]
        #print("Transpose",np.transpose(phi[x]))
        #ph1 = np.transpose(phi[x])
       # print("ph1",ph1)
        W2 = np.transpose(W1)
        #print("W2",W2)
        t2 = Y[x] - (W2.dot(ph1))
        #print("t2",t2)
        t3 = - (ph1.dot(t2))
        #print("t3: ",t3)
        #print("lamda: ",lamda)
        #de = dw / N
        ew = np.asarray(lamda)*W1
        #for i in W1:
         #   ew[i] = lamda*W1[i]
        #print("ew: ",ew)
        de = t3+ew
        #print("de: ",de)
        dw = (eta*de) / N
        #print("dw: ",dw)
        W2 = W1 - dw
      #  print("W2: ",W2)
    #    print("W2: ",W2)
        Erms2 = Ermscal(W2,X,Y,N,phi)
    #    print("ERMS2: ",Erms2)
       # if Erms2 - Erms1 <= 0.00001:
        #    break;
        Wt = W1
        W1 = W2
    #print("W2: ",W2)
    #print("ERMSREG2: ",Erms2)
   # print("ERMS REGULARized: ",Erms2)
   # print("ERMS REgularized Stochastic: ",Erms2)
        
    return W2
        
       
#LetoR(X,Y,Xvalid,Yvalid,Xtest,Ytest)

#----------- CSV Data Training ------------

X1 = np.loadtxt(open("Querylevelnorm_X.csv","rb"),delimiter=",")
#print("CSV X: ",X1)

Y1 = np.loadtxt(open("Querylevelnorm_t.csv","rb"),delimiter=",")
#print("CSV Y: ",Y1)

print("----------------Microsoft Real Time Data Set---------------")
LeToR1(X,Y,Xvalid,Yvalid,Xtest,Ytest,TrainN,valid,Test)

for row in range(0,TrainN):
    for col in range(0,46):
        X2[row][col]=X1[row][col]
        
for row in range(0,TrainN):
    #print("Y1[row]",Y1[row])
    Y2[row]=Y1[row]
    
xv=0
for row in range(TrainN, valid):
    for col in range(0,46):
        X2Valid[xv][col]=X1[row][col]
    xv=xv+1

yv=0
for row in range(TrainN,valid):
    Y2Valid[yv]=Y1[row]
        
xt=0
for row in range(ValidN, TestN):
    for col in range(0,46):
        X2Test[xt][col]=X1[row][col]
    xt=xt+1

yt=0
for row in range(ValidN, TestN):
    Y2Test[yt]=Y1[row]

print("")
print("----------------CSV Data Set---------------")
LeToR1(X2,Y2,X2Valid,Y2Valid,X2Test,Y2Test,TrainN,valid,Test)


