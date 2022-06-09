#include "fnn.h"
#include "matrix.h"

#define IMG_NUM 60000
#define IMG_SIZE 784
#define IMG_ROW 28
#define LABEL_SIZE 10


FNNForMNIST* create_fnn(int num_dense, int num_unit, double learning_rate){
    FNNForMNIST* fnn = (FNNForMNIST*)malloc(sizeof(FNNForMNIST));
    fnn->input_dim = IMG_SIZE;
    fnn->output_dim = LABEL_SIZE;
    fnn->hidden_dim = num_unit;
    fnn->hidden_layer_num = num_dense;
    fnn->lr = learning_rate;

    fnn->weights = (Matrix*)malloc(sizeof(Matrix)*(num_dense+1));
    fnn->bias = (double**)malloc(sizeof(double*)*(num_dense+1));
    fnn->bias_dim = (int*)malloc(sizeof(int)*(num_dense+1));

    // Initiate weights and bias for each layer with garbage values
    for (int i = 0; i < num_dense+1; i++){
        // w2: [input_dim, hidden_dim]
        if (i == 0){
            fnn->weights[i] = create_matrix(fnn->input_dim, fnn->hidden_dim);
            fnn->bias[i] = (double*)malloc(sizeof(double)*fnn->hidden_dim);
            fnn->bias_dim[i] = fnn->hidden_dim;
        }
        else if (i == num_dense){
            fnn->weights[i] = create_matrix(fnn->hidden_dim, fnn->output_dim);
            fnn->bias[i] = (double*)malloc(sizeof(double)*fnn->output_dim);
            fnn->bias_dim[i] = fnn->output_dim;
        }
        else{
            fnn->weights[i] = create_matrix(fnn->hidden_dim, fnn->hidden_dim);
            fnn->bias[i] = (double*)malloc(sizeof(double)*fnn->hidden_dim);
            fnn->bias_dim[i] = fnn->hidden_dim;
        }
    }

    // Allocate space for intermediate values
    fnn->activation = (Matrix*)malloc(sizeof(Matrix)*(fnn->hidden_layer_num+1));
    fnn->z = (Matrix*)malloc(sizeof(Matrix)*(fnn->hidden_layer_num+1));
    fnn->error = (Matrix*)malloc(sizeof(Matrix)*(fnn->hidden_layer_num));

    return fnn;
}

double step(FNNForMNIST* fnn, Matrix images, Matrix labels, int batch_size){
    // matrix_print(&images);
    double loss = 0.0;
    fnn->activation[0] = images;
    // labels = convert_to_matrix(labels);

    // Forward
    // Input layer
    Matrix hidden_states = matrix_dotp(fnn->activation[0], fnn->weights[0]); // [batch_size, hidden_dim]
    matrix_addbias(&hidden_states, fnn->bias[0]);
    fnn->z[0] = hidden_states;
    // matrix_print(&fnn->z[0]);
    matrix_sigmoid(&hidden_states);
    fnn->activation[1] = hidden_states;
    // matrix_print(&fnn->activation[1]);

    // Intermediate layers
    for (int i = 0; i < fnn->hidden_layer_num - 1; i++){
        hidden_states = matrix_dotp(hidden_states, fnn->weights[i + 1]);
        matrix_addbias(&hidden_states, fnn->bias[i + 1]);
        fnn->z[i + 1] = hidden_states;
        matrix_sigmoid(&hidden_states);
        fnn->activation[i + 2] = hidden_states;
    }

    // Output layer
    hidden_states = matrix_dotp(hidden_states, fnn->weights[fnn->hidden_layer_num]);
    matrix_addbias(&hidden_states, fnn->bias[fnn->hidden_layer_num]);
    fnn->z[fnn->hidden_layer_num] = hidden_states;
    matrix_sigmoid(&hidden_states);

    // Cost 
    Matrix diff = matrix_sub(hidden_states, labels);
    // matrix_abs(&diff);
    matrix_sigmoid_reverse(&fnn->z[fnn->hidden_layer_num]);
    fnn->error[0] = matrix_multiply(diff, fnn->z[fnn->hidden_layer_num]);

    loss = matrix_norm(&diff);

    // Back propogation
    for (int i = fnn->hidden_layer_num-1; i >= 0; i--){
        matrix_sigmoid_reverse(&fnn->z[i]);
        fnn->error[fnn->hidden_layer_num - i] = matrix_multiply(
            matrix_dotp(
                fnn->error[fnn->hidden_layer_num - i - 1],
                matrix_transpose(fnn->weights[i+1])
            ),
            fnn->z[i]
        );
    }

    // Gradient Descent
    for (int i = fnn->hidden_layer_num; i >= 0; i--){
        // Weights
        Matrix grad_w = matrix_dotp(
            matrix_transpose(fnn->activation[i]),
            fnn->error[fnn->hidden_layer_num - i]
        );
        matrix_scale(&grad_w, fnn->lr / (double)batch_size);
        fnn->weights[i] = matrix_sub(fnn->weights[i], grad_w);
        // Bias
        for (int j = 0; j < fnn->bias_dim[i]; j++){
            double error_sum = 0.0;
            for (int k = 0; k < batch_size; k++){
                error_sum += matrix_get_value(&fnn->error[fnn->hidden_layer_num - i], k, j);
            }
            fnn->bias[i][j] = fnn->bias[i][j] - fnn->lr / (double)batch_size * error_sum;
        }
    }

    return loss;
}

void free_fnn(FNNForMNIST* fnn){
    for (int i = 0; i < fnn->hidden_layer_num + 1; i++){
        matrix_free(&fnn->weights[i]);
    }
    free(fnn->weights);

    for (int i = 0; i < fnn->hidden_layer_num + 1; i++){
        free(&fnn->bias[i]);
    }
    free(fnn->bias);

    free(fnn->bias_dim);

    for (int i = 0; i < fnn->hidden_layer_num + 1; i++){
        matrix_free(&fnn->activation[i]);
    }
    free(fnn->activation);

    for (int i = 0; i < fnn->hidden_layer_num + 1; i++){
        matrix_free(&fnn->z[i]);
    }
    free(fnn->z);

    for (int i = 0; i < fnn->hidden_layer_num; i++){
        matrix_free(&fnn->error[i]);
    }
    free(fnn->error);
}

double step_prd(FNNForMNIST* fnn, Matrix images, Matrix labels, int batch_size){
    // matrix_print(&images);
    double loss;
    fnn->activation[0] = images;
    // labels = convert_to_matrix(labels);

    // Forward
    // Input layer
    Matrix hidden_states = matrix_dotp(fnn->activation[0], fnn->weights[0]); // [batch_size, hidden_dim]
    matrix_addbias(&hidden_states, fnn->bias[0]);
    fnn->z[0] = hidden_states;
    // matrix_print(&fnn->z[0]);
    matrix_sigmoid(&hidden_states);
    fnn->activation[1] = hidden_states;
    // matrix_print(&fnn->activation[1]);

    // Intermediate layers
    for (int i = 0; i < fnn->hidden_layer_num - 1; i++){
        hidden_states = matrix_dotp(hidden_states, fnn->weights[i + 1]);
        matrix_addbias(&hidden_states, fnn->bias[i + 1]);
        fnn->z[i + 1] = hidden_states;
        matrix_sigmoid(&hidden_states);
        fnn->activation[i + 2] = hidden_states;
    }

    // Output layer
    hidden_states = matrix_dotp(hidden_states, fnn->weights[fnn->hidden_layer_num]);
    matrix_addbias(&hidden_states, fnn->bias[fnn->hidden_layer_num]);
    fnn->z[fnn->hidden_layer_num] = hidden_states;
    matrix_sigmoid(&hidden_states);

    // Cost 
    Matrix diff = matrix_sub(hidden_states, labels);
    // 
    // // matrix_abs(&diff);
    // matrix_sigmoid_reverse(&fnn->z[fnn->hidden_layer_num]);
    // fnn->error[0] = matrix_multiply(diff, fnn->z[fnn->hidden_layer_num]);

    loss = matrix_norm(&diff);

    return loss;
}

