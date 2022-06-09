#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

// #ifndef mnist_header
// #define mnist_header
// #endif

struct MNISTDataset{
    int batch_size;
    int num_img;
    double** img_data;
    int* img_label;
    int sampled_num;
    int* indices;
    double** sampled_img;
    double** sampled_label;
    int* sampled_indices;
    int actual_batch_size;
};
typedef struct MNISTDataset MNISTDataset;

MNISTDataset* create_train_dataset(int batch_size);
MNISTDataset* create_test_dataset(int batch_size);

void get_batch_data(MNISTDataset* mnist, int total_num);
void save_img(double* image);
double** get_sampled_img(MNISTDataset* mnist);
double** get_sampled_label(MNISTDataset* mnist);
int get_actual_batch_size(MNISTDataset* mnist);
void print_sample(MNISTDataset* mnist);
void free_dataset(MNISTDataset* mnist);