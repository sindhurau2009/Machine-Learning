# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 17:05:08 2016

@author: sindhura
"""

import matplotlib.pyplot as plt
#----ERMS Closed form validation set----------
Eta = [[0.1],[0.2],[0.4],[0.5],[0.6],[0.7]]
Eval = [[47.9],[88.1],[90.1],[90.28],[89.67],[89.2]]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(Eta,Eval)
plt.xlabel('Eta')
plt.ylabel('Classification Error Rate')
plt.savefig('LR1.jpeg')


#...............SNN Eta.....................

Eta1 = [[0.01],[0.02],[0.03],[0.045],[0.05],[0.3]]
Eval1 = [[91.87],[90.6],[88.61],[85.582],[89.3],[12.8]]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(Eta1,Eval1)
plt.xlabel('Eta')
plt.ylabel('Classification Error Rate')
plt.savefig('SNN1.jpeg')


#....................SNN No of nodes.....................
Eta = [[100],[200],[300],[400],[500]]
Eval = [[89.3],[90.6],[88.61],[87.6],[91.87]]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(Eta,Eval)
plt.xlabel('Hidden Layer Nodes')
plt.ylabel('Classification Error Rate')
plt.savefig('SNN3.jpeg')

"""
#-------Regularized ERMS Closed form validation set--------
Mv = [[3],[4],[5],[6],[7],[9]]
ERMSv = [[0.349],[0.362],[0.505],[0.351],[0.452],[0.305]]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(Mv,ERMSv)
plt.xlabel('M with regularization Validation')
plt.ylabel('ERMS LETOR')
plt.savefig('images/MvEv2.jpeg')
#----------ERMS Closed form validation set CSV-------------
Mv = [[3],[4],[5],[6],[7],[9]]
ERMSv = [[0.178],[0.254],[0.280],[0.201],[0.453],[0.432]]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(Mv,ERMSv)
plt.xlabel('M')
plt.ylabel('ERMS Synthetic data')
plt.savefig('images/MvEv3.jpeg')
#-----------------Regularized ERMS Closed form validation set CSV---------
Mv = [[3],[4],[5],[6],[7],[9]]
ERMSv = [[0.174],[0.254],[0.280],[0.200],[0.453],[0.381]]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(Mv,ERMSv)
plt.xlabel('M with regularization')
plt.ylabel('ERMS Synthetic data')
plt.savefig('images/MvEv4.jpeg')
#-------------------ERMS Stochastic form validation set------------
Mv = [[3],[4],[5],[6],[7],[9]]
ERMSv = [[0.772],[0.827],[0.851],[0.841],[0.845],[0.741]]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(Mv,ERMSv)
plt.xlabel('M')
plt.ylabel('ERMS LETOR')
plt.savefig('images/MvEv5.jpeg')
#------------------Regularized ERMS Stochastic form validation set-------
Mv = [[3],[4],[5],[6],[7],[9]]
ERMSv = [[0.771],[0.813],[0.851],[0.841],[0.865],[0.740]]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(Mv,ERMSv)
plt.xlabel('M with regularization')
plt.ylabel('ERMS LETOR')
plt.savefig('images/MvEv6.jpeg')
#-----------------ERMS Stochastic form validation set CSV------------
Mv = [[3],[4],[5],[6],[7],[9]]
ERMSv = [[0.994],[0.998],[0.998],[0.998],[0.993],[0.996]]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(Mv,ERMSv)
plt.xlabel('M')
plt.ylabel('ERMS Synthetic data')
plt.savefig('images/MvEv7.jpeg')
#---------------Regularized ERMS Stochastic form validation set CSv ------------
Mv = [[3],[4],[5],[6],[7],[9]]
ERMSv = [[0.998],[0.998],[0.998],[0.998],[0.997],[0.996]]
fig1 = plt.figure()
ax1 = fig1.add_subplot(111)
ax1.scatter(Mv,ERMSv)
plt.xlabel('M with regularization')
plt.ylabel('ERMS Synthetic data')
plt.savefig('images/MvEv8.jpeg')
"""