# Fortran 77 Multilayer Perceptron
Authors: Nabhonil Kar (nkar@princeton.edu) & Francisco J. Tapiador (francisco.tapiador@uclm.es)

A multilayer perceptron ANN implementation in Fortran 77. The MNIST hand-written digit data set is used for benchmark.

## Requirements and Instructions
 - Have MNIST dataset in '/input' (already included).
 - Modify the training specifications in MLP_train.f.
 - Compile MLP_train.f and run to generate ANN weights in '/weights'. 
 - Modify the training specifications in MLP_test.f to match those of MLP_train.f.
 - Compile MLP_test.f and to generate results in '/output'.