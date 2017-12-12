/*
#########################################################
## Stat 202A - Homework 3
## Author: Sydney Kahmann
## Date : 10-25-17
## Description: This script implements QR decomposition,
## linear regression, and eigen decomposition / PCA 
## based on QR.
#########################################################
 
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
Sign function for later use 
~~~~~~~~~~~~~~~~~~~~~~~~~~~ */

// [[Rcpp::export()]]
double signC(double d){
  return d<0?-1:d>0? 1:0;
}



/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
Problem 1: QR decomposition 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */  


// [[Rcpp::export()]]
List myQRC(const mat A){ 
  
  /*
  Perform QR decomposition on the matrix A
  Input: 
  A, an n x m matrix (mat)
  
#############################################
## FILL IN THE BODY OF THIS FUNCTION BELOW ##
#############################################
  
  */ 
  
  int n = A.n_rows;
  int m = A.n_cols;
  mat R = A;
  mat Q = eye(n,n);
  
  for (int k=0; k < m; k++){
    vec  x = zeros<vec>(n);
    
    for(int j=k; j<x.n_elem; j++){ 
      x(j) = R(j,k); 
    }
    
    vec v=x;
    
    v(k)=x(k)+signC(x(k))*sqrt(sum(square(x)));
    
    double s=sqrt(sum(square(v)));
    vec u=v/s;
    R = R - 2*(u * u.t() * R);
    Q = Q - 2*(u * u.t() * Q);
    
  }
  
  List output;
  
  // Function should output a List 'output', with 
  // Q.transpose and R
  // Q is an orthogonal n x n matrix
  // R is an upper triangular n x m matrix
  // Q and R satisfy the equation: A = Q %*% R
  output["Q"] = Q.t();
  output["R"] = R;
  return(output);
  
}

/* ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
 Problem 2: Linear regression using QR 
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ */


// [[Rcpp::export()]]
mat myLinearRegressionC(const mat X, const mat Y){
  
  /*  
   Perform the linear regression of Y on X
   Input: 
   X is an n x p matrix of explanatory variables
   Y is an n dimensional vector of responses
   Do NOT simulate data in this function. n and p
   should be determined by X.
   Use myQRC inside of this function
   
   #############################################
   ## FILL IN THE BODY OF THIS FUNCTION BELOW ##
   #############################################
   
   */  
  
  mat XX=X;
  mat Ones = ones<mat>(X.n_rows,X.n_cols);
  XX.insert_cols(0,Ones);
  XX.shed_cols(0,Ones.n_cols-2);
  
  mat Z = join_rows(XX,Y);
  
  mat R = myQRC(Z)[1];
  
  int p = Z.n_cols-1;
   
  mat R1 = zeros<mat>(p,p);
  for(int r=0; r<p; r++){ 
    for(int q=0; q<p; q++){ 
      R1(r,q)= R(r,q);
    }
  }
  
  vec Y1 = zeros<vec>(p);
  for(int w=0; w<p; w++){ 
    Y1(w)=R(w,p);
  }
  
  vec beta_ls=solve(R1,Y1);

  // Function returns the 'p+1' by '1' matrix 
  // beta_ls of regression coefficient estimates
  return(beta_ls.t());
  
}  




/* ~~~~~~~~~~~~~~~~~~~~~~~~ 
Problem 3: PCA based on QR 
~~~~~~~~~~~~~~~~~~~~~~~~~~ */


// [[Rcpp::export()]]
List myEigen_QRC(const mat A, const int numIter = 1000){
  
  /*  
  
  Perform PCA on matrix A using your QR function, myQRC.
  Input:
  A: Square matrix
  numIter: Number of iterations
  
#############################################
## FILL IN THE BODY OF THIS FUNCTION BELOW ##
#############################################
  
  */  
  
  mat A_copy=A;
  int r = A.n_rows;
  int n = A.n_rows;
  int m = A.n_cols;
  int p = m;

  mat v= randn(r,r);      
  List e_QR;

  for(int i=0; i<numIter; i++){ 
    e_QR = myQRC(v);
    mat Q = e_QR["Q"];
    mat R = e_QR["R"];
    v = A_copy * Q;
  }
  
  mat Q = e_QR["Q"];
  mat R = e_QR["R"];
  
  vec D = diagvec(R);
  List output;
  
  // Function should output a list with D and V
  // D is a vector of eigenvalues of A
  // V is the matrix of eigenvectors of A (in the 
  // same order as the eigenvalues in D.)
  output["D"] = D.t();
  output["V"] = Q;
  return(output);
  
}
