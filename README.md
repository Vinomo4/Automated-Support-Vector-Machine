# Automated Support Vector Machine

> An automated version of an SVM (programmed in AMPL) using Python which includes primal/dual problem formulation and several methods for point generation

This project aims to create an SVM using AMPL coding both formulations of the problem (Primal and Dual) for the subsequent classification of points, the generation of which is performed randomly, to check the performance of the created SVM depending on the selected method of generation.

Another objective was allowing the user to perform the random point generation, the selection of the formulation of the problem and, after the classification is performed, the calculation of the accuracy achieved by the SVM automatically. To do so, a Python script was created that, using the AMPL models previously created, performs all these tasks in a user-friendly interface.

<p align="center">
  <img src='README Images/SVM_example.png'/>
</p>

## Usage

Firstly, the program will allow the user to select how to formulate the classification problem:

 * Primal SVM
 * Dual SVM
 * RBF SVM
 
 Once the formulation is selected, the user indicates which method wants to use for the generation of the random points:
 
 * Gensvmat
 * SwissRoll
 * [Diabetes datset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/diabetes)

*Note: If the diabetes dataset is not selected, the user must introduce the number of points to generate, the seed for the generation and the grade of penalization for the slacks (nu)*

Once the input is properly added by the user, the optimization problem is solved using AMPL and the training accuracy and parameters of interest are provided.

<p align="center">
  <img src='README Images/AMPL.PNG'/>
</p>

After the training phase, the program will now demand a new number of points and a new seed for the test values generation.
Just like with the training data, the obtained test accuracy will be shown to the user.

## Installation

In order to correctly use the programs provided, it is necessary to have installed the following libraries:

* `pip install numpy`

* `pip install sklearn`

* `pip install statistics`

* `pip install amplpy`
 
Also, the user must change in the [automatic.py](./automatic.py) file the *ampl_path* variable introducing the location of AMPL in his computer:

`ampl_path = '/user/directory/of/AMPL'`

## Architecture

The files contained on this repository are:

* [`Primal`](./SVM_Primal.mod) and [`Dual`](./SVM_Dual.mod) models of the SVM.
* Diabetes [`training`](./diabetes_train.dat) and [`test`](./diabetes_test.dat) datasets. (*Real dataset*)
* [Automatic Python interface](./automatic.py).
* [Gensvmdat](./gensvmdat)as one of the methods for random points generation.
* An extensive [report](./Report) about the development and conclusions of this project. (*Language: Catalan*)

*Note: The rest of the files in the master branch are auxiliary or license related*

## Team

This project was developed by:
| [![Vinomo4](https://avatars2.githubusercontent.com/u/49389601?s=60&v=4)](https://github.com/Vinomo4) | 
| --- | 
| [Victor Novelle Moriano](https://github.com/Vinomo4) | 


Student of Data Science and Engineering at [UPC](https://www.upc.edu/ca).

## License

[MIT License](./LICENSE)
