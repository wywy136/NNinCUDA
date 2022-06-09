#include "mnist.h"

#define IMG_NUM 60000
#define IMG_NUM_TEST 10000
#define IMG_SIZE 784
#define IMG_ROW 28
#define LABEL_SIZE 10


MNISTDataset* create_train_dataset(int batch_size){
    MNISTDataset* mnist = (MNISTDataset*)malloc(sizeof(MNISTDataset));
    mnist->batch_size = batch_size;
    mnist->num_img = IMG_NUM;

    // Allocate space for storing images
    mnist->img_data = (double **)malloc(IMG_NUM * sizeof(double*));
    for(int i = 0; i < IMG_NUM; i++){
        mnist->img_data[i] = (double *)malloc(IMG_SIZE * sizeof(double));
        memset(mnist->img_data[i], 0, sizeof(double) * IMG_SIZE);
    }

    // Allocate space for storing labels
    mnist->img_label = (int*)malloc(IMG_NUM * sizeof(int));

    FILE* img_file = fopen("/home/ywang27/HPC/project-4-wywy136/data/train-images-idx3-ubyte", "rb");
    int offset = 16;
    // A byte
    unsigned char buff[1];
    for (int i = 0; i < IMG_NUM; i++){
        for (int j = 0; j < IMG_SIZE; j++){
            fseek(img_file, offset, SEEK_SET);
            fread(buff, 1, 1, img_file);
            mnist->img_data[i][j] = (double)buff[0];
            offset += 1;
        }
    }
    fclose(img_file);
    // save_img(mnist->img_data[0]);

    FILE* img_label = fopen("/home/ywang27/HPC/project-4-wywy136/data/train-labels-idx1-ubyte", "rb");
    offset = 8;
    // A byte
    for (int i = 0; i < IMG_NUM; i++){
        fseek(img_label, offset, SEEK_SET);
        fread(buff, 1, 1, img_label);
        mnist->img_label[i] = (int)buff[0];
        offset += 1;
    }
    fclose(img_label);

    // Allocate space for indices
    mnist->indices = (int*)malloc(sizeof(int)*IMG_NUM);
    for (int i = 0; i < IMG_NUM; i++){
        mnist->indices[i] = i;
    }

    // Sampled num
    mnist->sampled_num = 0;
    mnist->sampled_indices = (int*)malloc(batch_size * sizeof(int));

    // Allocate space for sampling
    mnist->sampled_img = (double **)malloc(batch_size * sizeof(double*));
    for(int i = 0; i < batch_size; i++){
        mnist->sampled_img[i] = (double*)malloc(IMG_SIZE * sizeof(double));
        memset(mnist->sampled_img[i], 0, sizeof(double) * IMG_SIZE);
    }

    mnist->sampled_label = (double **)malloc(batch_size * sizeof(double*));
    for(int i = 0; i < batch_size; i++){
        mnist->sampled_label[i] = (double*)malloc(LABEL_SIZE * sizeof(double));
        memset(mnist->sampled_label[i], 0, sizeof(double) * LABEL_SIZE);
    }

    return mnist;
}

MNISTDataset* create_test_dataset(int batch_size){
    MNISTDataset* mnist = (MNISTDataset*)malloc(sizeof(MNISTDataset));
    mnist->batch_size = batch_size;
    mnist->num_img = IMG_NUM_TEST;

    // Allocate space for storing images
    mnist->img_data = (double **)malloc(IMG_NUM_TEST * sizeof(double*));
    for(int i = 0; i < IMG_NUM_TEST; i++){
        mnist->img_data[i] = (double *)malloc(IMG_SIZE * sizeof(double));
        memset(mnist->img_data[i], 0, sizeof(double) * IMG_SIZE);
    }

    // Allocate space for storing labels
    mnist->img_label = (int*)malloc(IMG_NUM_TEST * sizeof(int));

    FILE* img_file = fopen("/home/ywang27/HPC/project-4-wywy136/data/t10k-images-idx3-ubyte", "rb");
    int offset = 16;
    // A byte
    unsigned char buff[1];
    for (int i = 0; i < IMG_NUM_TEST; i++){
        for (int j = 0; j < IMG_SIZE; j++){
            fseek(img_file, offset, SEEK_SET);
            fread(buff, 1, 1, img_file);
            mnist->img_data[i][j] = (double)buff[0];
            // printf("%i ", mnist->img_label[i]);
            offset += 1;
        }
    }
    fclose(img_file);
    // save_img(mnist->img_data[0]);

    FILE* img_label = fopen("/home/ywang27/HPC/project-4-wywy136/data/t10k-labels-idx1-ubyte", "rb");
    offset = 8;
    // A byte
    for (int i = 0; i < IMG_NUM_TEST; i++){
        fseek(img_label, offset, SEEK_SET);
        fread(buff, 1, 1, img_label);
        mnist->img_label[i] = (int)buff[0];
        // printf("%i ", mnist->img_label[i]);
        offset += 1;
    }
    fclose(img_label);

    // Allocate space for indices
    mnist->indices = (int*)malloc(sizeof(int)*IMG_NUM_TEST);
    for (int i = 0; i < IMG_NUM_TEST; i++){
        mnist->indices[i] = i;
    }

    // Sampled num
    mnist->sampled_num = 0;
    mnist->sampled_indices = (int*)malloc(batch_size * sizeof(int));

    // Allocate space for sampling
    mnist->sampled_img = (double **)malloc(batch_size * sizeof(double*));
    for(int i = 0; i < batch_size; i++){
        mnist->sampled_img[i] = (double*)malloc(IMG_SIZE * sizeof(double));
        memset(mnist->sampled_img[i], 0, sizeof(double) * IMG_SIZE);
    }

    mnist->sampled_label = (double **)malloc(batch_size * sizeof(double*));
    for(int i = 0; i < batch_size; i++){
        mnist->sampled_label[i] = (double*)malloc(LABEL_SIZE * sizeof(double));
        memset(mnist->sampled_label[i], 0, sizeof(double) * LABEL_SIZE);
    }

    // for (int i = 0; i < batch_size; i++){
    //     for (int j = 0; j < LABEL_SIZE; j++){
    //         printf("%f ", mnist->sampled_label[i][j]);
    //     }
    //     printf("\n");
    // }

    return mnist;
}

void get_batch_data(MNISTDataset* mnist, int total_num){
    srand((unsigned)time(NULL));
    mnist->actual_batch_size = 0;
    for (int i = 0; i < mnist->batch_size; i++){
        // Copy img data
        int index = rand() % (total_num - mnist->sampled_num);
        int img_idx = mnist->indices[index];

        mnist->sampled_indices[i] = index;

        memcpy(mnist->sampled_img[i], mnist->img_data[img_idx], sizeof(double)*IMG_SIZE);
        // Set label data
        // memset(mnist->sampled_label[i], 0, sizeof(double) * LABEL_SIZE);
        mnist->sampled_label[i][mnist->img_label[img_idx]] = 1.0;
        // printf("%i\n", mnist->img_label[img_idx]);
        // for (int i = 0; i < mnist->batch_size; i++){
        //     for (int j = 0; j < LABEL_SIZE; j++){
        //         printf("%f ", mnist->sampled_label[i][j]);
        //     }
        //     printf("\n");
        // }
        // printf("%f\n", mnist->sampled_label[i][mnist->img_label[img_idx]]);
        // Swap
        int temp = mnist->indices[index];
        mnist->indices[index] = mnist->indices[total_num - mnist->sampled_num - 1];
        mnist->indices[total_num - mnist->sampled_num - 1] = temp;

        mnist->sampled_num += 1;
        mnist->actual_batch_size += 1;

        if (mnist->sampled_num == total_num){
            printf("All training data is exhausted. Starting a new iteration.\n");
            // Reset
            mnist->sampled_num = 0;
            for (int i = 0; i < total_num; i++){
                mnist->indices[i] = i;
            }
            return;
        }
    }
}

void print_sample(MNISTDataset* mnist){
    for (int i = 0; i < mnist->batch_size; i++){
        printf("%i ", mnist->sampled_indices[i]);
    }
    printf("\n");
}

double** get_sampled_img(MNISTDataset* mnist){
    return mnist->sampled_img;
}

double** get_sampled_label(MNISTDataset* mnist){
    return mnist->sampled_label;
}

int get_actual_batch_size(MNISTDataset* mnist){
    return mnist->actual_batch_size;
}

void save_img(double* img_data){
    FILE *fp = fopen("img.txt", "w");
    for (long i = 0; i < IMG_ROW; i++) {
        for (long j = 0; j < IMG_ROW; j++) {
            fprintf(fp, "%f ", img_data[i * IMG_ROW + j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void free_dataset(MNISTDataset* mnist){
    for(int i = 0; i < IMG_NUM; i++){
        free(mnist->img_data[i]);
    }
    free(mnist->img_data);

    free(mnist->img_label);

    free(mnist->sampled_indices);

    for(int i = 0; i < mnist->batch_size; i++){
        free(mnist->sampled_img[i]);
    }
    free(mnist->sampled_img);

    for(int i = 0; i < mnist->batch_size; i++){
        free(mnist->sampled_label[i]);
    }
    free(mnist->sampled_label);

    free(mnist->indices);
}