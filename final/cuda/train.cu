#include <cuda.h>

#include "mnist.h"
#include "fnn.h"
#include "matrix.h"
#include <stdlib.h>
#include <stdio.h>

#define IMG_SIZE 784
#define IMG_ROW 28
#define LABEL_SIZE 10
#define IMG_NUM 60000

#define NTHREADS_PER_BLOCK 4
#define NBLOCKS 4

// __global__
// void train(MNISTDataset* mnist_dataset, FNNForMNIST* fnn){
//     printf("Thread %d in block %d\n", threadIdx.x , blockIdx.x);

// }

int main(int argc, char* argv[]){
    int num_layers = strtol(argv[1], NULL, 10);
    int hidden_dim = strtol(argv[2], NULL, 10);
    int num_epoch = strtol(argv[3], NULL, 10);
    int batch_size = strtol(argv[4], NULL, 10);

    MNISTDataset* mnist_dataset = create_dataset(batch_size);
    printf("Loaded MNIST training set.\n");

    FNNForMNIST* fnn = create_fnn(num_layers, hidden_dim, 0.001);

    // MNISTDataset* mnist_dataset_device;
    // FNNForMNIST* fnn_device;

    // cudaMalloc((void**)&mnist_dataset_device, sizeof(MNISTDataset));
    // cudaMalloc((void**)&fnn_device, sizeof(FNNForMNIST));
    // cudaMemcpy(mnist_dataset_device, mnist_dataset, sizeof(MNISTDataset), cudaMemcpyHostToDevice);
    // cudaMemcpy(fnn_device, fnn, sizeof(FNNForMNIST), cudaMemcpyHostToDevice);

    // train<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(mnist_dataset_device, fnn_device);

    // cudaDeviceSynchronize();

    int batch_num = IMG_NUM / batch_size + 1;
    printf("Batch num: %i\n", batch_num);

    int actual_batch_size;

    // double loss;
    // double loss;

    // Matrix* fnn_weights = (Matrix*)malloc(sizeof(Matrix)*(num_layers+1));
    // Matrix* fnn_weights_device;


    for (int e = 0; e < 1; e++){
        for (int i = 0; i < 1; i++){
            get_batch_data(mnist_dataset);
            actual_batch_size = get_actual_batch_size(mnist_dataset);
            
            double* sampled_img = get_flat_sampled_img(mnist_dataset);
            double* sampled_label = get_flat_sampled_label(mnist_dataset);

            // Matrix* sampled_img_device;
            // Matrix* sampled_label_device;

            // cudaMalloc((void**)&sampled_img_device, sizeof(Matrix));
            // cudaMalloc((void**)&sampled_label_device, sizeof(Matrix));
            // cudaMemcpy(sampled_img_device, &sampled_img, sizeof(Matrix), cudaMemcpyHostToDevice);
            // cudaMemcpy(sampled_label_device, &sampled_label, sizeof(Matrix), cudaMemcpyHostToDevice);

            double loss = step(fnn, sampled_img, sampled_label, actual_batch_size, NBLOCKS, NTHREADS_PER_BLOCK);
            // step_gpu<<<NBLOCKS, NTHREADS_PER_BLOCK>>>(fnn_device, sampled_img_device, sampled_label_device, actual_batch_size);
            // cudaDeviceSynchronize();
            // if (i % 500 == 0){
            //     // print_sample(mnist_dataset);
            //     printf("Epoch: [%i/%i]\tBatch: [%i/%i]\tLoss: %f\n", e+1, num_epoch, i+1, batch_num, loss);
            // }
        }
    }

    // cudaFree(mnist_dataset_device);
    // cudaFree(fnn_device);

    // free_dataset(mnist_dataset);
    // free_fnn(fnn);
}
