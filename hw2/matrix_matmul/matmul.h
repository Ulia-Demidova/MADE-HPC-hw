#ifndef MATRIX_MATMUL_MATMUL_H
#define MATRIX_MATMUL_MATMUL_H
#include <iostream>

void RandomMatrix(double *A, size_t n);

void RandomVector(double *vec, size_t n);

double MatMatMul(double *A, double *B, double *C, size_t n);
double MatVecMul(double *A, double *v, double*w, size_t n);
#endif //MATRIX_MATMUL_MATMUL_H
