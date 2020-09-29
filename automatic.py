from amplpy import AMPL,Environment
import os
import numpy as np
import statistics
from sklearn import datasets as ds

#Global variables (The paths must be changed by the user).
#---------------------------------------------------------
#Folder where AMPL is located on the computer.
ampl_path = '/home/victor/ampl_linux-intel64'

#If the .py is saved in the same folder of gensvmdat, change path to "".
#Otherwise put the correct folder path and the new files will be stored there.
path = '/home/victor/Escritorio/Matlab/OM/ProjecteSVM'

#Epsilon used to avoid problems of precision with floating points.
#It is recommended to not change this value.
epsilon = 1e-05
#---------------------------------------------------------
#Functions that allow the set-up the connection between python and AMPL.
ampl = AMPL(Environment('/home/victor/ampl_linux-intel64'))
#If another solver than cplex wants to be used, modify that parameter with
#the name of the desired solver.
ampl.setOption('solver', 'cplex')
#---------------------------------------------------------
#Auxiliar function to generate data with the swiss function.
def swiss_generation(num_points,seed,test):
    A,y = ds.make_swiss_roll(num_points,2,seed)
    #aux = statistics.median(y)
    aux = np.mean(y)
    for i in range(num_points):
        if(y[i] < aux): y[i] = -1
        else: y[i] = 1

    wr = np.matrix(A); ite = np.matrix(y); ite = ite.reshape(ite.size,1); write = np.c_[wr,ite];

    if (test): word = "test"
    else: word = "train"

    with open(path+"/clean_data_"+word+".dat","w") as clean:
        for line in write:
            np.savetxt(clean, line,fmt = "%f")
    return A,y

#---------------------------------------------------------
#Function that generates the training input for the problem.
def Input(num_points,seed,nu_value,method,p_type,test):

    if(test): word = "test"
    else: word = "train"
    #Gensvmdat
    if (p_type == 1):
        os.system(path+"/gensvmdat data_"+word+".dat "+str(num_points)+" "+str(seed))

        with open(path+"/data_"+word+".dat","r") as raw, \
             open(path+"/clean_data_"+word+".dat","w") as clean:
                data = raw.read()
                data = data.replace('*','').replace('   ',' ')
                clean.write(data)

        os.remove(path+"/data_"+word+".dat")

        A = np.loadtxt(path+"/clean_data_"+word+".dat", delimiter=' ')
        y = A[:,A[0].size-1]
        A = np.delete(A,A[0].size-1,1)

    #Swiss_roll
    elif(p_type == 2): A,y = swiss_generation(num_points,seed,test)

    else:

        A = np.loadtxt(path+"/diabetes_"+word+".dat", delimiter=' ')
        y = A[:,A[0].size-1]
        A = np.delete(A,A[0].size-1,1)

    if(test):
        print("\nInput test created!\n")
        return A,y

    #Primal problem.
    if(method == 1):
        aux = A[0].size
        spaces = "     "
        var = "A"
        val = A

    #Dual problem.
    elif(method != 1):
        if(method == 2): K = np.dot(A,A.T)
        #RBF
        else:
            K = np.zeros((num_points,num_points))
            s2 = np.mean(np.var(A,0))
            for i in range(num_points):
                for j in range(i,num_points):
                    K[i,j] = K[j,i] = np.exp(- np.linalg.norm(A[i,:] - A[j,:])**2/(2*s2))

        K = np.matrix(K+ np.eye(num_points)*epsilon)
        aux = K[:,0].size
        spaces = " "
        var = "K"
        val = K

    o = open(path+"/input_train.dat","w")
    o.write("param m := "+str(num_points)+";\n")
    if(method == 1): o.write("param n := "+str(aux)+";\n")
    o.write("param nu:= "+str(nu_value)+";\n\n")
    o.write("param "+var+" :\n    ")
    for i in range (1,aux+1):
        o.write(str(i)+spaces)
    o.write(":=\n")
    for i in range(num_points):
        o.write(str(i+1)+" ")
        o.write(str(val[i,:]).replace('[','').replace(']','')+"\n")
    o.write(";\n\nparam y :=\n")
    for i in range(num_points):
        o.write(str(i+1) + " " + str(y[i]) +"\n")
    o.write(";")

    print("\nInput created!\n")

    return A,y
#---------------------------------------------------------
#Function that returns the necessary parameters and variables depending on the method of solving.
def Param_and_Var(method,num_points,p_type):

    y = ampl.getParameter('y').getValues()
    y = np.matrix(y.toPandas())
    y = y.reshape(y.size,1);

    if(p_type == 1): col = 4
    elif(p_type == 2): col = 3
    else: col = 8

    #Primal problem.
    if(method == 1):

        w = ampl.getVariable('w').getValues()
        w = np.matrix(w.toPandas())
        w = w.reshape(w.size, 1)

        A = ampl.getParameter('A').getValues()
        A = np.matrix(A.toPandas())
        A = A.reshape(num_points,col)

        gamma = ampl.getVariable('gamma').getValues()
        gamma = np.matrix(gamma.toPandas())
        gamma = gamma.reshape(gamma.size,1);

        return y,w,A,gamma

    #Dual problem
    else :

        landa = ampl.getVariable('landa').getValues()
        landa = np.matrix(landa.toPandas())
        landa = landa.reshape(landa.size, 1)

        nu = ampl.getParameter('nu').getValues()
        nu = np.matrix(nu.toPandas())
        nu = nu.reshape(1,1)

        K = ampl.getParameter('K').getValues()
        K = np.matrix(K.toPandas())
        K = K.reshape(num_points,num_points)

        if(method ==2):
            if(p_type != 3):
                A = np.loadtxt(path+"/clean_data_train.dat", delimiter=' ')
            else:
                A = np.loadtxt(path+"/diabetes_train.dat", delimiter=' ')
            A = np.delete(A,A[0].size-1,1)
            return y,landa,nu,K,A

        return y,landa,nu,K
#---------------------------------------------------------
#Function that computes the values of w in the Dual problem.
def find_w(y,landa,A,num_points):
    w = [0 for i in range(A[0].size)]

    for i in range (num_points):
        aux = landa[i]*y[i]*A[i,:]
        w += aux

    w = np.matrix(w)
    w = w.reshape(w.size, 1)

    return w
#---------------------------------------------------------
#Function that computes the values of gamma in the Dual problem.
def find_gamma(y,landa,K,nu,num_points):
    #In order to obtain the gamma, first we find a SV point.

    SV = 0
    for i in range (num_points):
        if (epsilon < landa[i] < nu-epsilon):
                SV = i;
                break;

    #Then we proceed to calculate the gamma.

    gamma = y[SV]

    for j in range(num_points):
        gamma = gamma -landa[j]*y[j]*K[SV,j]

    return gamma

#---------------------------------------------------------
#Function that classifies the points according to the computed SVM.
def classify(method,num_points,gamma,*args):

    classification = [0 for i in range(num_points)]

    #Primal problem.
    if(method == 1):
        A = args[0]; w = args[1];

        for i in range(num_points):
            if(np.dot(A[i,:],w) + gamma > 0): classification[i] = 1
            else: classification[i] = -1

    #Dual problems.
    else:

        landa = args[0]; y = args[1]; K = args[2];

        for i in range(num_points):
            aux = 0
            for j in range(num_points):
                aux = aux + landa[j]*y[j]*K[j,i]
            if(aux+gamma > 0): classification[i] = 1
            else: classification[i] = -1

    return classification
#---------------------------------------------------------
#Function that computes the accuracy of the SVM.
def accuracy(classification,num_points,y):
    misclassified = 0
    for j in range(num_points):
        if(classification[j] != y[j]):misclassified += 1

    return (1-misclassified/num_points)*100
#---------------------------------------------------------
def SVM(method,num_points,p_type):

    if(method == 1): model = path+'/SVM_Primal.mod';
    else: model = path+'/SVM_Dual.mod';

    data = path+'/input_train.dat';

    ampl.reset()
    ampl.read(model)
    ampl.readData(data)

    ampl.solve()

    if(method == 1):  y,w,A,gamma = Param_and_Var(method,num_points,p_type)
    else:
        if(method == 2):
            y,landa,nu,K,A = Param_and_Var(method,num_points,p_type)
            w = find_w(y,landa,A,num_points)
        else: y,landa,nu,K = Param_and_Var(method,num_points,p_type)
        gamma = find_gamma(y,landa,K,nu,num_points)

    if(method != 3):
        print("\nValues of the weigths:\n")
        print(w)

    print("\nValue of gamma:\n")
    print(gamma)

    if(method == 1): classification = classify(method,num_points,gamma,A,w)
    else: classification = classify(method,num_points,gamma,landa,y,K)

    acc = accuracy(classification,num_points,y)

    print("\nAccuracy of the training classification:", acc,"%.\n")

    if(method != 3): return gamma,w
    return gamma,landa

#---------------------------------------------------------
def accuracy_test(num_points_test,seed_test,method,gamma,w,p_type):

    Obs_test,y_test = Input(num_points_test,seed_test,0,method,p_type,True)

    classification_test = classify(1,num_points_test,gamma,Obs_test,w)

    acc_t = accuracy(classification_test,num_points_test,y_test)

    print("\nAccuracy of the test classification:", acc_t,"%.\n")
#---------------------------------------------------------
def accuracy_test_RBF(num_points_test,seed_test,gamma,p_type,num_points_train,A_tr,y_tr,landa):

    Obs_test,y_test = Input(num_points_test,seed_test,0,3,p_type,True)

    s2 = np.mean(np.var(A_tr, 0))

    classification = [0 for i in range(num_points_test)]

    for i in range(num_points_test):
        aux = 0
        for j in range(num_points_train):
            K = np.exp(- np.linalg.norm(Obs_test[i,:] - A_tr[j,:])**2/(2*s2))
            aux = aux + landa[j]*y_tr[j]*K

        if(aux+gamma > 0): classification[i] = 1
        else: classification[i] = -1

    acc_t = accuracy(classification,num_points_test,y_test)

    print("\nAccuracy of the test classification:", acc_t,"%.\n")
#---------------------------------------------------------
def main():

    while True:
        print("Please introduce which SVM problem you want to solve:")
        print("1.(Primal SVM) - 2.(Dual SVM) - 3.(RBF SVM)")
        try :
            method = int(input())
            if (1 <= method <= 3): break;
        except: continue;

    while True:
        print("Now introduce which method for the point generation you want to use:")
        print("1.(Gensvmdat) - 2.(Swiss roll) - 3.(Diabetes data set)")
        try :
            p_type = int(input())
            if (1 <= p_type <= 3): break;
        except: continue;

    while True:
        print("Now introduce the parameters for the points generation.")
        try:
            if(p_type != 3):
                print("Usage: (Number of points) - (Seed) - (Value of nu)")
                num_points_train,seed_train,nu_value  = input().split()
                num_points_train = int(num_points_train); seed_train = int(seed_train); nu_value = float(nu_value);
            else:
                print("Usage: (Value of nu)")
                nu_value = float(input())
                num_points_train = 512
                seed_train = 0
            break;
        except: continue;

    A_tr,y_tr = Input(num_points_train,seed_train,nu_value,method,p_type,False)

    if (method != 3):
        gamma, w = SVM(method,num_points_train,p_type)

    else: gamma,landa = SVM(method,num_points_train,p_type)
    while True:
        try:
            if(p_type != 3):
                print("Now introduce the parameters for the test points generation.")
                print("Usage: (Number of points) - (Seed)")
                num_points_test,seed_test = input().split()
                num_points_test = int(num_points_test); seed_test = int(seed_test);
            else:
                num_points_test = 256
                seed_test  = 0
            break;
        except: continue;

    if(method != 3): accuracy_test(num_points_test,seed_test,method,gamma,w,p_type)
    else: accuracy_test_RBF(num_points_test,seed_test,gamma,p_type,num_points_train,A_tr,y_tr,landa)

    if(p_type != 3): os.remove(path+"/clean_data_train.dat")

if __name__ == "__main__":
    main()
