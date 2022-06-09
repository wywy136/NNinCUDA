#include <cuda.h>

#include "fnn.h"
#include "matrix.h"

#define IMG_NUM 60000
#define IMG_SIZE 784
#define IMG_ROW 28
#define LABEL_SIZE 10


__host__
FNNForMNIST* create_fnn(int num_dense, int num_unit, double learning_rate){
    FNNForMNIST* fnn = (FNNForMNIST*)malloc(sizeof(FNNForMNIST));
    fnn->input_dim = IMG_SIZE;
    fnn->output_dim = LABEL_SIZE;
    fnn->hidden_dim = num_unit;
    fnn->hidden_layer_num = num_dense;
    fnn->lr = learning_rate;

    // double * a = malloc
    // double a[num_unit]

    // fnn->weights = (Matrix*)malloc(sizeof(Matrix)*(num_dense+1));
    fnn->weights = (double**)malloc(sizeof(double*)*(num_dense+1));
    // for (int i = 0; i < num_dense + 1; i++){
    //     fnn->weights[i] = (double*)malloc(sizeof(double)*())
    // }
    fnn->bias = (double**)malloc(sizeof(double*)*(num_dense+1));
    fnn->bias_dim = (int*)malloc(sizeof(int)*(num_dense+1));

    // Initiate weights and bias for each layer with garbage values
    for (int i = 0; i < num_dense+1; i++){
        // w2: [input_dim, hidden_dim]
        if (i == 0){
            // fnn->weights[i] = create_matrix(fnn->input_dim, fnn->hidden_dim);
            fnn->weights[i] = (double*)malloc(sizeof(double)*(fnn->input_dim * fnn->hidden_dim));
            fnn->bias[i] = (double*)malloc(sizeof(double)*fnn->hidden_dim);
            fnn->bias_dim[i] = fnn->hidden_dim;
        }
        else if (i == num_dense){
            // fnn->weights[i] = create_matrix(fnn->hidden_dim, fnn->output_dim);
            fnn->weights[i] = (double*)malloc(sizeof(double)*(fnn->output_dim * fnn->hidden_dim));
            fnn->bias[i] = (double*)malloc(sizeof(double)*fnn->output_dim);
            fnn->bias_dim[i] = fnn->output_dim;
        }
        else{
            // fnn->weights[i] = create_matrix(fnn->hidden_dim, fnn->hidden_dim);
            fnn->weights[i] = (double*)malloc(sizeof(double)*(fnn->hidden_dim * fnn->hidden_dim));
            fnn->bias[i] = (double*)malloc(sizeof(double)*fnn->hidden_dim);
            fnn->bias_dim[i] = fnn->hidden_dim;
        }
    }

    // Allocate space for intermediate values
    // fnn->activation = (Matrix*)malloc(sizeof(Matrix)*(fnn->hidden_layer_num+1));
    fnn->activation = (double**)malloc(sizeof(double*) * (fnn->hidden_layer_num+1));
    fnn->z = (double**)malloc(sizeof(double*) * (fnn->hidden_layer_num+1));
    fnn->error = (double**)malloc(sizeof(double*) * (fnn->hidden_layer_num));

    return fnn;
}

__host__
double step(FNNForMNIST* fnn, double* images, double* labels, int batch_size, int nblocks, int nthreads_per_block){
    // printf("Thread %d in block %d\tRow: %d\tCol: %d\n", threadIdx.x , blockIdx.x, blockIdx.x*blockDim.x+threadIdx.x, blockIdx.y*blockDim.y+threadIdx.y);
    double loss = 0.0;
    fnn->activation[0] = images;
    // // labels = convert_to_matrix(labels);

    // Forward
    // Input layer
    double* hidden_states;   // ******
    cudaMalloc((void**)&hidden_states, sizeof(double) * batch_size * fnn->hidden_dim);
    double* hidden_states_host = (double*)malloc(sizeof(double) * batch_size * fnn->hidden_dim);  // ******
    memset(hidden_states_host, 0, sizeof(double) * batch_size * fnn->hidden_dim);
    cudaMemcpy(hidden_states, hidden_states_host, sizeof(double) * batch_size * fnn->hidden_dim, cudaMemcpyHostToDevice);

    double* act;   // ******
    cudaMalloc((void**)&act, sizeof(double) * batch_size * IMG_SIZE);
    cudaMemcpy(act, fnn->activation[0], sizeof(double) * batch_size * IMG_SIZE, cudaMemcpyHostToDevice);
    
    double* w;  // ******
    cudaMalloc((void**)&w, sizeof(double) * fnn->hidden_dim * fnn->hidden_dim);
    cudaMemcpy(w, fnn->weights[0], sizeof(double) * fnn->input_dim * fnn->hidden_dim, cudaMemcpyHostToDevice);
    
    flat_matrix_dotp<<<nblocks, nthreads_per_block>>>(
        hidden_states, 
        act, 
        w, 
        batch_size,
        fnn->hidden_dim,
        fnn->input_dim
    ); // -> [batch_size, hidden_dim]
    cudaDeviceSynchronize();

    cudaMemcpy(hidden_states_host, hidden_states, sizeof(double) * batch_size * fnn->hidden_dim, cudaMemcpyDeviceToHost);
    printf("H[0][0]: %f", hidden_states_host[0]);

    flat_matrix_addbias(hidden_states_host, batch_size, fnn->hidden_dim, fnn->bias[0]);
    
    // Set z[0]
    fnn->z[0] = (double*)malloc(sizeof(double) * batch_size * fnn->hidden_dim);
    for (int i = 0; i < batch_size * fnn->hidden_dim; i++){
        fnn->z[0][i] = hidden_states_host[i];
    }

    flat_matrix_sigmoid(hidden_states_host, batch_size, fnn->hidden_dim);
    
    // Set activation[1]
    fnn->activation[1] = (double*)malloc(sizeof(double) * batch_size * fnn->hidden_dim);
    for (int i = 0; i < batch_size * fnn->hidden_dim; i++){
        fnn->z[0][i] = hidden_states_host[i];
    }

    // Intermediate layers
    for (int i = 0; i < fnn->hidden_layer_num - 1; i++){
        cudaMemcpy(hidden_states, hidden_states_host, sizeof(double) * batch_size * fnn->hidden_dim, cudaMemcpyHostToDevice);
        cudaMemcpy(w, fnn->weights[i + 1], sizeof(double) * fnn->hidden_dim * fnn->hidden_dim, cudaMemcpyHostToDevice);
        flat_matrix_dotp<<<nblocks, nthreads_per_block>>>(
            hidden_states, 
            hidden_states, 
            w,
            batch_size,
            fnn->hidden_dim,
            fnn->hidden_dim
        );
        cudaDeviceSynchronize();

        cudaMemcpy(hidden_states_host, hidden_states, sizeof(double) * batch_size * fnn->hidden_dim, cudaMemcpyDeviceToHost);

        flat_matrix_addbias(hidden_states_host, batch_size, fnn->hidden_dim, fnn->bias[i + 1]);

        fnn->z[i + 1] = (double*)malloc(sizeof(double) * batch_size * fnn->hidden_dim);
        for (int j = 0; j < batch_size * fnn->hidden_dim; j++){
            fnn->z[i + 1][j] = hidden_states_host[j];
        }

        flat_matrix_sigmoid(hidden_states_host, batch_size, fnn->hidden_dim);

        fnn->activation[i + 2] = (double*)malloc(sizeof(double) * batch_size * fnn->hidden_dim);
        for (int j = 0; j < batch_size * fnn->hidden_dim; j++){
            fnn->activation[i+2][j] = hidden_states_host[j];
        }
    }

    // Output layer
    cudaMemcpy(hidden_states, hidden_states_host, sizeof(double) * batch_size * fnn->hidden_dim, cudaMemcpyHostToDevice);
    cudaMemcpy(w, fnn->weights[fnn->hidden_layer_num], sizeof(double) * fnn->hidden_dim * fnn->output_dim, cudaMemcpyHostToDevice);
    // cudaFree(hidden_states);
    // cudaMalloc((void**)&hidden_states, sizeof(double) * batch_size * fnn->output_dim);
   
    double* hidden_states_output;   // ******
    cudaMalloc((void**)&hidden_states_output, sizeof(double) * batch_size * fnn->output_dim);
    flat_matrix_dotp<<<nblocks, nthreads_per_block>>>(
        hidden_states_output, 
        hidden_states, 
        w,
        batch_size,
        fnn->hidden_dim,
        fnn->output_dim
    );
    cudaDeviceSynchronize();

    // free(hidden_states_host);
    double* hidden_states_output_host = (double*)malloc(sizeof(double) * batch_size * fnn->output_dim);
    
    cudaMemcpy(hidden_states_output_host, hidden_states_output, sizeof(double) * batch_size * fnn->output_dim, cudaMemcpyDeviceToHost);
    // cudaFree(hidden_states_output);
    
    flat_matrix_addbias(hidden_states_output_host, batch_size, fnn->output_dim, fnn->bias[fnn->hidden_layer_num]);
    fnn->z[fnn->hidden_layer_num] = (double*)malloc(sizeof(double) * batch_size * fnn->output_dim);
    for (int i = 0; i < batch_size * fnn->output_dim; i++){
        fnn->z[fnn->hidden_layer_num][i] = hidden_states_output_host[i];
    }
    
    flat_matrix_sigmoid(hidden_states_output_host, batch_size, fnn->output_dim);

    // Cost
    flat_matrix_sub(hidden_states_output_host, hidden_states_host, labels, batch_size, fnn->output_dim);
    loss = flat_matrix_norm(hidden_states_output_host, batch_size, fnn->output_dim);

    flat_matrix_sigmoid_reverse(fnn->z[fnn->hidden_layer_num], batch_size, fnn->output_dim);
    flat_matrix_multiply(hidden_states_output_host, hidden_states_output_host, fnn->z[fnn->hidden_layer_num], batch_size, fnn->output_dim);
    // Set error[0]
    double* err = (double*)malloc(sizeof(double)*batch_size*fnn->output_dim);
    for (int i = 0; i < batch_size*fnn->output_dim; i++){
        // fnn->error[0][i] = hidden_states_output_host[i];
        err[i] = hidden_states_output_host[i];
    }
    // fnn->error[0] = err;

    // Back propogation
    for (int i = fnn->hidden_layer_num-1; i >= 0; i--){
        flat_matrix_sigmoid_reverse(fnn->z[i], batch_size, fnn->hidden_dim);
        // fnn->error[fnn->hidden_layer_num - i] = matrix_multiply(
        //     matrix_dotp(
        //         fnn->error[fnn->hidden_layer_num - i - 1],
        //         matrix_transpose(fnn->weights[i+1])
        //     ),
        //     fnn->z[i]
        // );
    }

    // // Gradient Descent
    // for (int i = fnn->hidden_layer_num; i >= 0; i--){
    //     // Weights
    //     Matrix grad_w = matrix_dotp(
    //         matrix_transpose(fnn->activation[i]),
    //         fnn->error[fnn->hidden_layer_num - i]
    //     );
    //     matrix_scale(&grad_w, fnn->lr / (double)batch_size);
    //     fnn->weights[i] = matrix_sub(fnn->weights[i], grad_w);
    //     // Bias
    //     for (int j = 0; j < fnn->bias_dim[i]; j++){
    //         double error_sum = 0.0;
    //         for (int k = 0; k < batch_size; k++){
    //             error_sum += matrix_get_value(&fnn->error[fnn->hidden_layer_num - i], k, j);
    //         }
    //         fnn->bias[i][j] = fnn->bias[i][j] - fnn->lr / (double)batch_size * error_sum;
    //     }
    // }

    return loss;
}

// void free_fnn(FNNForMNIST* fnn){
//     for (int i = 0; i < fnn->hidden_layer_num + 1; i++){
//         matrix_free(&fnn->weights[i]);
//     }
//     free(fnn->weights);

//     for (int i = 0; i < fnn->hidden_layer_num + 1; i++){
//         free(&fnn->bias[i]);
//     }
//     free(fnn->bias);

//     free(fnn->bias_dim);

//     for (int i = 0; i < fnn->hidden_layer_num + 1; i++){
//         matrix_free(&fnn->activation[i]);
//     }
//     free(fnn->activation);

//     for (int i = 0; i < fnn->hidden_layer_num + 1; i++){
//         matrix_free(&fnn->z[i]);
//     }
//     free(fnn->z);

//     for (int i = 0; i < fnn->hidden_layer_num; i++){
//         matrix_free(&fnn->error[i]);
//     }
//     free(fnn->error);
// }