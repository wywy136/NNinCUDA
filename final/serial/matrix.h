#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

struct Matrix{
    int row;
    int col;
    double** data;
};

typedef struct Matrix Matrix;

Matrix create_matrix(int row, int col);

Matrix matrix_dotp(Matrix a, Matrix b);

Matrix matrix_add(Matrix a, Matrix b);

Matrix matrix_sub(Matrix a, Matrix b);

Matrix matrix_multiply(Matrix a, Matrix b);

Matrix matrix_transpose(Matrix m);

Matrix convert_to_matrix(double** a, int row, int col);

void matrix_sigmoid(Matrix* a);

void matrix_sigmoid_reverse(Matrix* a);

void matrix_addbias(Matrix* m, double* b);

void matrix_abs(Matrix* m);

void matrix_scale(Matrix* m, double s);

double matrix_norm(Matrix* m);

double matrix_get_value(Matrix*m, int row, int col);

void matrix_print(Matrix* m);

void matrix_size(Matrix* m);

void matrix_free(Matrix* m);

void matrix_tofile(Matrix* m, char* filename);