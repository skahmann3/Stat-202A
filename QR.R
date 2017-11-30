#########################################################
## Stat 202A - Homework 3
## Author: Sydney Kahmann
## Date : 10 - 25- 17
## Description: This script implements QR decomposition,
## linear regression, and eigen decomposition / PCA 
## based on QR.
#########################################################

#############################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names, 
## function inputs or outputs. You can add examples at the
## end of the script (in the "Optional examples" section) to 
## double-check your work, but MAKE SURE TO COMMENT OUT ALL 
## OF YOUR EXAMPLES BEFORE SUBMITTING.
##
## Very important: Do not use the function "setwd" anywhere
## in your code. If you do, I will be unable to grade your 
## work since R will attempt to change my working directory
## to one that does not exist.
#############################################################

##################################
## Function 1: QR decomposition ##
##################################

myQR <- function(A){
  
  ## Perform QR decomposition on the matrix A
  ## Input: 
  ## A, an n x m matrix
  
  ########################
  ## FILL IN CODE BELOW ##
  ########################  
  
  n = dim(A)[1]
  m = dim(A)[2]
  R=A
  Q = diag(n)
  
  for (k in 1:m){
    x = rep(0,n) 
    x[k:n]= R[k:n,k]
    v=x
    
    v[k]=x[k]+sign(x[k])*sqrt(sum(x^2))
    s=sqrt(sum(v^2))
    
    u=v/s
    R = R - 2*(u %*% t(u) %*% R)
    Q = Q - 2*(u %*% t(u) %*% Q)

  }

  ## Function should output a list with Q.transpose and R
  ## Q is an orthogonal n x n matrix
  ## R is an upper triangular n x m matrix
  ## Q and R satisfy the equation: A = Q %*% R
  return(list("Q" = t(Q), "R" = R))
  
}


###############################################
## Function 2: Linear regression based on QR ##
###############################################

myLM <- function(X, Y){
  
  ## Perform the linear regression of Y on X
  ## Input: 
  ## X is an n x p matrix of explanatory variables
  ## Y is an n dimensional vector of responses
  ## Do NOT simulate data in this function. n and p
  ## should be determined by X.
  ## Use myQR inside of this function
  
  ########################
  ## FILL IN CODE BELOW ##
  ########################  
  
  Z <- cbind(rep(1,nrow(X)), X,Y)
  R <- myQR(Z)$R
  
  p=ncol(Z)-1
  
  R1=R[1:p,1:p]
  Y1=R[1:p,p+1]
  beta_ls=solve(R1,Y1)

  ## Function returns the 1 x (p + 1) vector beta_ls, 
  ## the least squares solution vector
  return(beta_ls)
  
}

##################################
## Function 3: PCA based on QR  ##
##################################

myEigen_QR <- function(A, numIter = 1000){
  
  ## Perform PCA on matrix A using your QR function, myQR.
  ## Input:
  ## A: Square matrix
  ## numIter: Number of iterations
  
  ########################
  ## FILL IN CODE BELOW ##
  ######################## 
  
  A_copy=A
  r = dim(A_copy)[1]
  n = dim(A)[1]
  m = dim(A)[2]
  p = m
  v= matrix(rnorm(n * p), nrow = r)
  
  for (i in 1:numIter){
    e_QR = myQR(v)
    v = A_copy %*% e_QR$Q
  }

  ## Function should output a list with D and V
  ## D is a vector of eigenvalues of A
  ## V is the matrix of eigenvectors of A (in the 
  ## same order as the eigenvalues in D.)
  return(list("D" = diag(e_QR$R), "V" = e_QR$Q))
  
}


########################################################
## Optional examples (comment out before submitting!) ##
########################################################

# testing_Linear_Regression <- function(){
# 
#   ## This function is not graded; you can use it to
#   ## test out the 'myLinearRegression' function
# 
#   ## Define parameters
#   n    <- 10
#   p    <- 5
# 
#   ## Simulate data from our assumed model.
#   ## We can assume that the true intercept is 0
#   X    <- matrix(rnorm(n * p), nrow = n)
#   beta <- matrix(1:p, nrow = p)
#   Y    <- X %*% beta + rnorm(n)
# 
#   ## Save R's linear regression coefficients
#   R_coef  <- coef(lm(Y ~ X))
# 
#   ## Save our linear regression coefficients
#   my_coef <- myLM(X, Y)
# 
#   ## Are these two vectors different?
#   sum_square_diff <- sum((R_coef - my_coef)^2)
#   if(sum_square_diff <= 0.001){
#     return('Both results are identical')
#   }else{
#     return('There seems to be a problem...')
#   }
# 
# }
# 
# testing_Linear_Regression()
# 
# 
# A = matrix(sample(1:50,25), nrow=5)
# A= A+t(A)
#   
# eigen(A)
#   
# myEigen_QR(A)
