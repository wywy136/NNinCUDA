#pragma once
#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "matrix.h"


struct FNNForMNIST{
    double** weights;
    double** bias;
    int* bias_dim;
    int input_dim;
    int output_dim;
    int hidden_dim;
    int hidden_layer_num;
    double lr;

    double** activation;
    double** z;
    double** error;
};
typedef struct FNNForMNIST FNNForMNIST;

FNNForMNIST* create_fnn(int num_dense, int num_unit, double learning_rate);

double step(FNNForMNIST* fnn, double* images, double* labels, int batch_size, int nblocks, int nthreads_per_block);
void free_fnn(FNNForMNIST* fnn);