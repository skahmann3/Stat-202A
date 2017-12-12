#########################################################
## Stat 202A - Final Project
## Author: Sydney Kahmann
## Date : 12-11-17
## Description: This script implements a two layer neural network
#########################################################

#############################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names,
## function inputs or outputs. You can add examples at the
## end of the script (in the "Optional examples" section) to
## double-check your work, but MAKE SURE TO COMMENT OUT ALL
## OF YOUR EXAMPLES BEFORE SUBMITTING.
##
## Very important: Do not use the function "os.chdir" anywhere
## in your code. If you do, I will be unable to grade your
## work since Python will attempt to change my working directory
## to one that does not exist.
#############################################################
import numpy as np
import csv
import random
import math
import matplotlib.pyplot as plt

#import os
#os.chdir('C:/Users/Sydney/OneDrive/2017-18/202A/Homework/')


def prepare_data(valid_digits=np.array((6,5))):
    if len(valid_digits)!=2:
        raise Exception("Error: you must specify exactly 2 digits for classification!")

    csvfile=open('digits.csv','r')
    reader=csv.reader(csvfile)
    data=[]
    for line in reader:
        data.append(line)

    csvfile.close()
    digits=np.asarray(data,dtype='float')

    X=digits[(digits[:,64]==valid_digits[0]) | (digits[:,64]==valid_digits[1]),0:64]
    Y = digits[(digits[:, 64] == valid_digits[0]) | (digits[:, 64] == valid_digits[1]), 64:65]

    X=np.asarray(map(lambda k: X[k,:]/X[k,:].max(), range(0,len(X))))

    Y[Y==valid_digits[0]]=0
    Y[Y==valid_digits[1]]=1

    training_set=random.sample(range(360),270)

    testing_set=list(set(range(360)).difference(set(training_set)))

    X_train=X[training_set,:]
    Y_train=Y[training_set,]

    X_test=X[testing_set,:]
    Y_test=Y[testing_set,]

    return X_train,Y_train,X_test,Y_test


def accuracy(p,y):

    acc=np.mean((p>0.5)==(y==1))
    return acc

####################################################
## Two Layer Neural Network  ##
####################################################

## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
## Train a two layer neural network to classify the digits data ##
## Use Relu as the activation function for the first layer. Use Sigmoid as the activation function for the second layer##
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ##
def my_NN(X_train,Y_train,X_test,Y_test,num_hidden=20,num_iterations=1000,learning_rate=1e-1):

    n=X_train.shape[0]
    p=X_train.shape[1]+1
    ntest=X_test.shape[0]

    X_train1= np.concatenate((np.repeat(1,n,axis=0).reshape((n,1)),X_train),axis=1)
    X_test1 = np.concatenate((np.repeat(1, ntest, axis=0).reshape((ntest, 1)), X_test), axis=1)
    alpha=np.random.standard_normal((p,num_hidden))
    beta=np.random.standard_normal((num_hidden+1,1))
    acc_train=np.repeat(0.,num_iterations)
    acc_test=np.repeat(0.,num_iterations)

    #######################
    ## FILL IN CODE HERE ##
    #######################

    for it in range(num_iterations):
        Z = 1 / (1 + np.exp(np.dot(-X_train1, alpha)))
        Z1 = np.hstack((np.repeat(1., n, axis=0).reshape((n,1)), Z))
        pr = 1 / (1 + np.exp(np.dot(-Z1, beta)))        
        m=Z1.shape[1]
        dbeta = np.dot(np.repeat(1., n, axis=0).reshape((1,n)), (np.repeat(Y_train-pr, m).reshape(n, m)*Z1)/n)
        beta = beta + learning_rate * np.transpose(dbeta)
        
        for k in range(num_hidden):
            da = (Y_train - pr)*beta[k+1]*Z[:, k].reshape(n,1)*(1-Z[:, k].reshape(n,1))
            dalpha = np.dot(np.repeat(1., n, axis=0).reshape((1, n)), (np.repeat(da, p).reshape(n, p)*X_train1)/n)
            alpha[:, [k]] = (alpha[:, k].reshape(p,1) + (learning_rate * np.transpose(dalpha))).reshape(p,1)
        
        acc_train[it] = accuracy(pr, Y_train)
        Ztest = 1/(1 + np.exp(np.dot(-X_test1, alpha)))
        Ztest1 = np.hstack((np.repeat(1., ntest, axis=0).reshape((ntest,1)), Ztest))
        prtest = 1/(1 + np.exp(np.dot(-Ztest1, beta)))
        acc_test[it] = accuracy(prtest, Y_test)
        print "On iteration ", it, " the training accuracy is ", acc_train[it], " and the testing accuracy is ", acc_test[it]

    ## Function should output 4 things:
    ## 1. The learned parameters of the first layer of the neural network, alpha
    ## 2. The learned parameters of the second layer of the neural network, beta
    ## 3. The accuracy over the training set, acc_train (a "num_iterations" dimensional vector).
    ## 4. The accuracy over the testing set, acc_test (a "num_iterations" dimensional vector).
    return alpha,beta,acc_train,acc_test


############################################################################
## Test your functions and visualize the results here##
############################################################################
"""
X_train, Y_train, X_test, Y_test = prepare_data()
alpha,beta,acc_train,acc_test=my_NN(X_train,Y_train,X_test,Y_test,num_hidden=50,num_iterations=1000,learning_rate=1e-2)

X_train, Y_train, X_test, Y_test = prepare_data()
x = np.arange(start=1, stop=1001)
plt.plot(x, acc_train)
plt.plot(x, acc_test)
plt.legend(['acc_train', 'acc_test'], loc='upper left')
plt.show()
"""

####################################################
## Optional examples (comment out your examples!) ##
####################################################




