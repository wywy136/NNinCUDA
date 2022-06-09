#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "matrix.h"


struct FNNForMNIST{
    Matrix* weights;
    double** bias;
    int* bias_dim;
    int input_dim;
    int output_dim;
    int hidden_dim;
    int hidden_layer_num;
    double lr;

    Matrix* activation;
    Matrix* z;
    Matrix* error;
};
typedef struct FNNForMNIST FNNForMNIST;

FNNForMNIST* create_fnn(int num_dense, int num_unit, double learning_rate);

double step(FNNForMNIST* fnn, Matrix images, Matrix labels, int batch_size);
double step_prd(FNNForMNIST* fnn, Matrix images, Matrix labels, int batch_size);
void free_fnn(FNNForMNIST* fnn);

// __global__
// void step_gpu(FNNForMNIST* fnn, Matrix* images, Matrix* labels, int batch_size);