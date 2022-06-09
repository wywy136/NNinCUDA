#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>

struct Matrix{
    int row;
    int col;
    double** data;
};

typedef struct Matrix Matrix;

Matrix create_matrix(int row, int col);

__global__
void flat_matrix_dotp(double* c, double* a, double* b, int row, int col, int dim);

double* matrix_flatten(Matrix m);

Matrix matrix_add(Matrix a, Matrix b);

void flat_matrix_sub(double* c, double* a, double* b, int row, int col);

void flat_matrix_multiply(double* c, double* a, double* b, int row, int col);

Matrix matrix_transpose(Matrix m);

Matrix convert_to_matrix(double** a, int row, int col);

void flat_matrix_sigmoid(double* a, int row, int col);

void flat_matrix_sigmoid_reverse(double* a, int row, int col);

void flat_matrix_addbias(double* m, int row, int col, double* b);

void matrix_abs(Matrix* m);

void matrix_scale(Matrix* m, double s);

double flat_matrix_norm(double* m, int row, int col);

double matrix_get_value(Matrix*m, int row, int col);

void matrix_print(Matrix* m);

void matrix_size(Matrix* m);

void matrix_free(Matrix* m);

int matrix_get_row(Matrix m);

int matrix_get_col(Matrix m);