#include "matrix.h"


Matrix create_matrix(int row, int col){
    Matrix m;
    m.row = row;
    m.col = col;
    m.data = (double**)malloc(sizeof(double*) * row);
    for (int i = 0; i < row; i++){
        m.data[i] = (double*)malloc(sizeof(double) * col);
        // memset(m.data[i], 1.0, sizeof(double)*col);
        for (int j = 0; j < col; j++){
            m.data[i][j] = 0.0;
        }
    }
    return m;
}

Matrix convert_to_matrix(double** a, int row, int col){
    Matrix m = create_matrix(row, col);
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            m.data[i][j] = a[i][j];
        }
    }
    return m;
}

double* matrix_flatten(Matrix m){
    double* data_1d = (double*)malloc(sizeof(double) * m.row * m.col);
    for (int i = 0; i < m.row; i++){
        for (int j = 0; j < m.col; j++){
            data_1d[i * m.row + j] = m.data[i][j];
        }
    }
    return data_1d;
}

int matrix_get_row(Matrix m){
    return m.row;
}

int matrix_get_col(Matrix m){
    return m.col;
}

__global__
void flat_matrix_dotp(double* c, double* a, double* b, int row, int col, int dim){
    printf("Thread %d in block %d\n", threadIdx.x , blockIdx.x);
    // if (threadIdx.x == 0){
    //     c[0] = 1.0;
    // }
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int target_row = index / col;
    int target_col = index % col;

    for (int i = 0; i < dim; i++){
        c[target_row * col + target_col] = a[target_row * dim + i] * b[i * dim + col];
    }

    // Matrix c = create_matrix(row, col);
    // for (int i = 0; i < row; i++){
    //     for (int j = 0; j < col; j++){
    //         for (int k = 0; k < dim; k++){
    //             c.data[i][j] += a.data[i][k] * b.data[k][j];
    //         }
    //     }
    // }
    // return c;
}

Matrix matrix_add(Matrix a, Matrix b){
    int row = a.row;
    int col = b.col;
    Matrix c = create_matrix(row, col);
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            c.data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }
    return c;
}

void flat_matrix_sub(double* c, double* a, double* b, int row, int col){
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            c[i * row + j] = a[i * row + j] - b[i * row + j];
        }
    }
}

void flat_matrix_multiply(double* c, double* a, double* b, int row, int col){
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            c[i * row + j] = a[i * row + j] * b[i * row + j];
        }
    }
}

void flat_matrix_sigmoid(double* m, int row, int col){
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            m[i * row + j] = 1.0 / (1.0 + exp(-m[i * row + j]));
        }
    }
}

void flat_matrix_sigmoid_reverse(double* m, int row, int col){
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            double sig = 1.0 / (1.0 + exp(-m[i * row + j]));
            m[i * row + j] = sig * (1 - sig);
        }
    }
}

void flat_matrix_addbias(double* m, int row, int col, double* b){
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            m[i*row + j] += b[j];
        }
    }
}

void matrix_abs(Matrix* m){
    for (int i = 0; i < m->row; i++){
        for (int j = 0; j < m->col; j++){
            m->data[i][j] = fabs(m->data[i][j]);
        }
    }
}

Matrix matrix_transpose(Matrix m){
    int row = m.row;
    int col = m.col;
    Matrix t = create_matrix(col, row);
    for (int i = 0; i < col; i++){
        for (int j = 0; j < row; j++){
            t.data[i][j] = m.data[j][i];
        }
    }
    return t;
}

void matrix_scale(Matrix* m, double s){
    for (int i = 0; i < m->row; i++){
        for (int j = 0; j < m->col; j++){
            m->data[i][j] = s * m->data[i][j];
        }
    }
}

double flat_matrix_norm(double* m, int row, int col){
    double norm = 0.0;
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            norm += m[i * row + j] * m[i * row + j];
        }
    }
    return sqrt(norm);
}

double matrix_get_value(Matrix* m, int row, int col){
    return m->data[row][col];
}

void matrix_print(Matrix* m){
    for (int i = 0; i < m->row; i++){
        for (int j = 0; j < m->col; j++){
            printf("%f\t", m->data[i][j]);
        }
        printf("\n");
    }
}

void matrix_size(Matrix* m){
    printf("[%i, %i]\n", m->row, m->col);
}

void matrix_free(Matrix* m){
    for (int i = 0; i < m->row; i++){
        free(m->data[i]);
    }
    free(m->data);
}
