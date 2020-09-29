# Parameters
param m >= 1 integer;
param nu >= 0;

param y {i in 1..m};
param K {1..m,1..m};

#Variables
var landa {i in 1..m} >= 0,<= nu;

#SVM Dual.

maximize SVM_Dual: sum{i in 1..m}(landa[i]) - 1/2*(sum{i in 1..m}(sum{z in 1..m}(landa[i]*y[i]*landa[z]*y[z]*K[i,z])));
subject to Restriccion: sum{i in 1..m}(landa[i]*y[i]) = 0;
