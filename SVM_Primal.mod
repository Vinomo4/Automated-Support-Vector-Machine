# Parameters
param n >= 1, integer;
param m >= 1, integer;
param nu >= 0;

param y {i in 1..m};
param A {1..m,1..n};

#Variables
var gamma;
var w {i in 1..n};
var s {i in 1..m} >= 0;

# SVM primal.

minimize SVM_Primal :
	1/2*sum{i in 1..n}(w[i]*w[i]) + nu*sum{j in 1..m}(s[j]);

subject to Restriccion{i in 1..m}:
	y[i]*(sum{j in 1..n}(A[i,j]*w[j]) + gamma) +s[i] >= 1;
