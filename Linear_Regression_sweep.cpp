/*
####################################################
## Stat 202A - Homework 2
## Author: Sydney Kahmann 
## Date : 10-16-17
## Description: This script implements linear regression 
## using the sweep operator
####################################################
 
###########################################################
## INSTRUCTIONS: Please fill in the missing lines of code
## only where specified. Do not change function names, 
## function inputs or outputs. MAKE SURE TO COMMENT OUT ALL 
## OF YOUR EXAMPLES BEFORE SUBMITTING.
##
## Very important: Do not change your working directory
## anywhere inside of your code. If you do, I will be unable 
## to grade your work since R will attempt to change my 
## working directory to one that does not exist.
###########################################################
 
 */ 


# include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]
using namespace Rcpp;
using namespace arma;


/* ~~~~~~~~~~~~~~~~~~~~~~~~~ 
Problem 1: Sweep operator 
~~~~~~~~~~~~~~~~~~~~~~~~~ */

// [[Rcpp::export()]]
mat mySweepC(const mat A, int m){
  
  /*
  Perform a SWEEP operation on A with the pivot element A[m,m].
  
  A: a square matrix (mat).
  m: the pivot element is A[m, m]. 
  Returns a swept matrix B (which is m by m).
  
  Note the "const" in front of mat A; this is so you
  don't accidentally change A inside your code.
  
#############################################
## FILL IN THE BODY OF THIS FUNCTION BELOW ##
#############################################
  */
  
  mat B = A;
  int n = B.n_rows;
  
  double d = 1;
  
  for(int k=0; k < m; k++){
    
    d = d*B(k,k);
    
    for(int r =0; r <n; r++){
      for(int j=0; j<n; j++){
        if(r != k && j != k)
          B(r,j) = B(r,j) - B(r,k)*B(k,j)/B(k,k);
      }
    }
    
    for(int i=0; i<n; i++){
      if( i != k)
        B(i,k) = B(i,k)/B(k,k);
    }
    
    for(int j=0; j<n; j++){
      if( j != k)
        B(k,j) = B(k,j)/B(k,k);
    }
    
    B(k,k) = -1/B(k,k);
    
  }
  
  // Return swept matrix B
  return(B);
  
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
 Problem 2: Linear regression using the sweep operator 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

// [[Rcpp::export()]]
mat myLinearRegressionC(const mat X, const mat Y){

/*  
Find the regression coefficient estimates beta_hat
corresponding to the model Y = X * beta + epsilon
Your code must use the sweep operator you coded above.
Note: we do not know what beta is. We are only 
given a matrix X and a matrix Y and we must come 
up with an estimate beta_hat.

X: an 'n row' by 'p column' matrix of input variables.
Y: an 'n row' by '1 column' matrix of responses

#############################################
## FILL IN THE BODY OF THIS FUNCTION BELOW ##
#############################################
*/  

// Let me start things off for you...
  
  mat XX=X;
  
  mat Ones = ones<mat>(X.n_rows,X.n_cols);
  
  XX.insert_cols(0,Ones);
  XX.shed_cols(0,Ones.n_cols-2);

  int n = XX.n_rows;
  int p = XX.n_cols;
  
  mat Z = join_rows(XX,Y);
  
  mat A = trans(Z) * Z;
  int m = p;
  
  mat B = mySweepC(A,m);
  
  mat BH = B.rows(0,p-1);
  
  int g = B.n_cols;
  
  vec beta_hat = BH.col(g-1);

// Function returns the 'p+1' by '1' matrix 
// beta_hat of regression coefficient estimates
return(beta_hat);

}

