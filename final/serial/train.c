#include "mnist.h"
#include "fnn.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define IMG_SIZE 784
#define IMG_ROW 28
#define LABEL_SIZE 10
#define IMG_NUM 60000
#define IMG_NUM_TEST 10000

int main(int argc, char* argv[]){
    int num_layers = strtol(argv[1], NULL, 10);
    int hidden_dim = strtol(argv[2], NULL, 10);
    int num_epoch = strtol(argv[3], NULL, 10);
    int batch_size = strtol(argv[4], NULL, 10);

    MNISTDataset* mnist_train_dataset = create_train_dataset(batch_size);
    printf("Loaded MNIST training set.\n");

    // MNISTDataset* mnist_test_dataset = create_test_dataset(batch_size);
    // printf("Loaded MNIST test set.\n");

    FNNForMNIST* fnn = create_fnn(num_layers, hidden_dim, 0.01);

    int batch_num = IMG_NUM / batch_size + 1;
    printf("Batch num: %i\n", batch_num);

    int actual_batch_size;
    double loss;

    clock_t start,end;

    start = clock();
    for (int e = 0; e < num_epoch; e++){
        for (int i = 0; i < batch_num; i++){
            get_batch_data(mnist_train_dataset, IMG_NUM);
            actual_batch_size = get_actual_batch_size(mnist_train_dataset);
            
            Matrix sampled_img = convert_to_matrix(get_sampled_img(mnist_train_dataset), batch_size, IMG_SIZE);
            Matrix sampled_label = convert_to_matrix(get_sampled_label(mnist_train_dataset), batch_size, LABEL_SIZE);

            loss = step(fnn, sampled_img, sampled_label, actual_batch_size);
            
            if (i % 500 == 0){
                printf("Epoch: [%i/%i]\tBatch: [%i/%i]\tLoss: %f\n", e+1, num_epoch, i+1, batch_num, loss);
            }
        }
    }
    end = clock();
    printf("Elapsed time=%f\n", (double)(end-start)/1000000);
    printf("Grind rate: %f\n", num_epoch * IMG_NUM / ((double)(end-start)/1000000));

    // printf("Evaluating...\n");
    // batch_num = IMG_NUM_TEST / batch_size + 1;
    // for (int i = 0; i < batch_num; i++){
    //     get_batch_data(mnist_test_dataset, IMG_NUM_TEST);
    //     actual_batch_size = get_actual_batch_size(mnist_test_dataset);
        
    //     Matrix sampled_img = convert_to_matrix(get_sampled_img(mnist_test_dataset), batch_size, IMG_SIZE);
    //     Matrix sampled_label = convert_to_matrix(get_sampled_label(mnist_test_dataset), batch_size, LABEL_SIZE);

    //     loss = step_prd(fnn, sampled_img, sampled_label, actual_batch_size);
        
    //     if (i % 200 == 0){
    //         print_sample(mnist_test_dataset);
    //         printf("Batch: [%i/%i]\tLoss: %f\n", i+1, batch_num, loss);
    //     }
    // }
}

