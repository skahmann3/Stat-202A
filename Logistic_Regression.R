#########################################################
## Stat 202A - Homework 4
## Author: Sydney Kahmann
## Date : 10-31-17
## Description: This script implements logistic regression
## using iterated reweighted least squares using the code 
## we have written for linear regression based on QR 
## decomposition
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


################ Optional ################ 
## If you are using functions from Rcpp, 
## uncomment and adjust the following two 
## lines accordingly. Make sure that your 
## cpp file is in the current working
## directory, so you do not have to specify
## any directories in sourceCpp(). For 
## example, do not write
## sourceCpp('John_Doe/Homework/Stats202A/QR.cpp')
## since I will not be able to run this.

## library(Rcpp)
## sourceCpp(name_of_cpp_file)

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
  
  for (k in 1:(m-1)){
    x = rep(0,n) 
    x[k:n]= R[k:n,k]
    v=x
    
    v[k]=x[k]+sign(x[k])*norm(x,type="2")

    u=v/norm(v,type="2")
    R = R - 2*u %*% t(u) %*% R
    Q = Q - 2*u %*% t(u) %*% Q
    
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
  ## Use myQR (or myQRC) inside of this function
  
  ########################
  ## FILL IN CODE BELOW ##
  ########################  
  
  Z <- cbind(X,Y)
  R <- myQR(Z)$R
  
  p=ncol(X)
  n=nrow(X)
  
  R1=R[1:p,1:p]
  Y1=R[1:p,p+1]
  beta_ls=solve(R1 )%*% Y1
  
  ## Function returns the least squares solution vector
  return(beta_ls)
  
}

######################################
## Function 3: Logistic regression  ##
######################################

## Expit/sigmoid function
expit <- function(x){
  1 / (1 + exp(-x))
}

myLogistic <- function(X, Y){
  
  ## Perform the logistic regression of Y on X
  ## Input: 
  ## X is an n x p matrix of explanatory variables
  ## Y is an n dimensional vector of binary responses
  ## Use myLM (or myLMC) inside of this function
  
  ########################
  ## FILL IN CODE BELOW ##
  ########################
  
  r= dim(X)[1]
  c= dim(X)[2]
  
  Xcopy=X
  Ycopy=Y
  
  betas <- rep(0,c)
  epsilon = 10^(-6)
  
  err=10
  while(err>epsilon){
    
    eta <- Xcopy %*% betas
    p <- expit(eta)
    
    w= p *(1-p)
    z=eta+(Ycopy-p)/w
    sw=sqrt(w)
    mw=rep(sw,c)
    
    x_new=mw*Xcopy
    y_new=sw*z
    
    new_beta <- myLM(x_new, y_new)
    err=sum(abs(new_beta-betas))
    betas=new_beta
    
  }
  
  ## Function returns the logistic regression solution vector
  t(betas)  
  
}


## Optional testing (comment out!)
# n <- 100
# p <- 4
# 
# X    <- matrix(rnorm(n * p), nrow = n)
# beta <- rnorm(p)
# Y    <- 1 * (runif(n) < expit(X %*% beta))
# 
# logistic_beta <- myLogistic(X, Y)
# logistic_beta
# glm(Y~X+0, family=binomial)
