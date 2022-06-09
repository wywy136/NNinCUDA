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

int matrix_get_row(Matrix m){
    return m.row;
}

int matrix_get_col(Matrix m){
    return m.col;
}

Matrix matrix_dotp(Matrix a, Matrix b){
    int row = a.row;
    int col = b.col;
    int dim = b.row;
    Matrix c = create_matrix(row, col);
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            for (int k = 0; k < dim; k++){
                c.data[i][j] += a.data[i][k] * b.data[k][j];
            }
        }
    }
    return c;
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

Matrix matrix_sub(Matrix a, Matrix b){
    int row = a.row;
    int col = b.col;
    Matrix c = create_matrix(row, col);
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            c.data[i][j] = a.data[i][j] - b.data[i][j];
        }
    }
    return c;
}

Matrix matrix_multiply(Matrix a, Matrix b){
    int row = a.row;
    int col = b.col;
    Matrix c = create_matrix(row, col);
    for (int i = 0; i < row; i++){
        for (int j = 0; j < col; j++){
            c.data[i][j] = a.data[i][j] * b.data[i][j];
        }
    }
    return c;
}

void matrix_sigmoid(Matrix* a){
    for (int i = 0; i < a->row; i++){
        for (int j = 0; j < a->col; j++){
            a->data[i][j] = 1.0 / (1.0 + exp(-a->data[i][j]));
        }
    }
}

void matrix_sigmoid_reverse(Matrix* a){
    for (int i = 0; i < a->row; i++){
        for (int j = 0; j < a->col; j++){
            double sig = 1.0 / (1.0 + exp(-a->data[i][j]));
            a->data[i][j] = sig * (1 - sig);
        }
    }
}

void matrix_addbias(Matrix* m, double* b){
    for (int i = 0; i < m->row; i++){
        for (int j = 0; j < m->col; j++){
            m->data[i][j] += b[j];
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

double matrix_norm(Matrix* m){
    double norm = 0.0;
    for (int i = 0; i < m->row; i++){
        for (int j = 0; j < m->col; j++){
            norm += m->data[i][j] * m->data[i][j];
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

void matrix_tofile(Matrix* m, char* filename){
    FILE *fp = fopen(filename, "w");

    for (int i = 0; i < m->row; i++) {
        for (long j = 0; j < m->col; j++) {
            fprintf(fp, "%.9le ", m->data[i][j]);
        }
        fprintf(fp, "\n");
        printf("%f ", m->data[i][200]);
    }
    fclose(fp);
}