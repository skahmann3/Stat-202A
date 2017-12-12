# -*- coding: utf-8 -*-
"""

 Stat 202A - Homework 2
 Author: Sydney Kahmann
 Date : 10-17-17
 Description: This script implements linear regression 
 using the sweep operator

 INSTRUCTIONS: Please fill in the missing lines of code
 only where specified. Do not change function names, 
 function inputs or outputs. You can add examples at the
 end of the script (in the "Optional examples" section) to 
 double-check your work, but MAKE SURE TO COMMENT OUT ALL 
 OF YOUR EXAMPLES BEFORE SUBMITTING.

 Do not use any of Python's built in functions for matrix 
 inversion or for linear modeling (except for debugging or 
 in the optional examples section).
 
"""

import numpy as np
#from sklearn import linear_model

################################
## Function 1: Sweep operator ##
################################

def mySweep(A, m):
    
    """
    Perform a SWEEP operation on A with the pivot element A[m,m].
    
    :param A: a square matrix (np.array).
    :param m: the pivot element is A[m, m].
    :returns a swept matrix (np.array). Original matrix is unchanged.
    
    FILL IN THE BODY OF THIS FUNCTION BELOW
    """
    
    B = np.copy(A)   
    n = B.shape[0]
    
    d=1
    
    for k in range(m):
        d = d*B[k,k]
    
        for i in range(n):
            for j in range(n):
                if i != k and j!= k:
                    B[i,j] = B[i,j] - B[i,k]*B[k,j]/B[k,k]
          
        for i in range(n):
            if i != k:
                B[i,k] = B[i,k]/B[k,k]
    
    
        for j in range(n):
            if j != k:
                B[k,j] = B[k,j]/B[k,k]

        B[k,k] = -1/B[k,k]
      
    ## The function outputs the matrix (np.array) B
    return(B)
  



########################################################
## Function 2: Linear regression using Sweep operator ##
########################################################

def myLinearRegression(X, Y):
  
  """
  Find the regression coefficient estimates beta_hat
  corresponding to the model Y = X * beta + epsilon
  Your code must use the sweep operator you coded above.
  Note: we do not know what beta is. We are only 
  given a matrix X and a vector Y and we must come 
  up with an estimate beta_hat.
  
  X: an 'n row' by 'p column' matrix (np.array) of input variables.
  Y: an n-dimensional vector (np.array) of responses

  FILL IN THE BODY OF THIS FUNCTION BELOW
  """
  
  ## Let me start things off for you...
  n = X.shape[0]
  p = X.shape[1]
  
  
  Z = np.hstack((np.ones((n, 1)), X, Y.reshape(n,1)))
  A = np.dot(np.transpose(Z), Z)
  S = mySweep(A, p+1)

  ## Function returns the (p+1)-dimensional vector 
  ## beta_hat of regression coefficient estimates
  beta_hat = S[range(p+1), p+1]
  
  
  ## Function returns the (p+1)-dimensional vector (np.array) 
  ## beta_hat of regression coefficient estimates
  return beta_hat
  


########################################################
## Optional examples (comment out before submitting!) ##
########################################################

#def testing_Linear_Regression():
#  
#  # This function is not graded; you can use it to 
#  # test out the 'myLinearRegression' function 
#
#  # You can set up a similar test function as was 
#  # provided to you in the R file.
#  
#   n=100
#   p=3
#   X= np.random.randn(n,p)
#   beta = np.array([[1],[2],[3]])
#   Y= np.dot(X, beta)+np.random.randn(n,1)
#   my_coef = myLinearRegression(X,Y)
##
### Create a linear regression object
#   regr = linear_model.LinearRegression()
##
### Fit the linear regression to our data
#   regr.fit(X, Y)
##
### Print model coefficients and intercept
#   # print "Intercept: \n", regr.intercept_
#   # print "Coefficients: \n", regr.coef_
#   
#   C_coef = np.append([regr.intercept_], [[regr.coef_]])
#      
#   ## Are these two vectors different?
#   sum_square_diff = sum((C_coef - my_coef)**2)
#   if sum_square_diff <= 0.001:
#     return('Both results are identical')
#   else:
#     return('There seems to be a problem...')
#   
# 
#testing_Linear_Regression()
