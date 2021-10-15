#include "matmul.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

void ZeroVector(double *v, size_t N) {
  for(size_t i = 0; i < N; i++) {
    v[i] = 0.0;
  }
}

void RandomVector(double *v, size_t N) {
  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    v[i] = rand() / RAND_MAX;
  }
}

void ZeroMatrix(double *A, size_t N) {
  for(size_t i = 0; i < N; i++) {
    for(size_t j = 0; j < N; j++) {
      A[i * N + j] = 0.0;
    }
  }
}

void RandomMatrix(double *A, size_t N) {
  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < N; j++) {
      A[i * N + j] = rand() / RAND_MAX;
    }
  }
}


double MatMatMul(double *A, double *B, double *C, size_t N) {
  struct timeval start, end;
  double r_time = 0.0;
  size_t i, j, k;

  size_t dummy = 0;

  ZeroMatrix(C, N);

  gettimeofday(&start, NULL);

  for (k = 0; k < N; k++) {
    for (i = 0; i < N; i++) {
      dummy = i * N;
      for (j = 0; j < N; j++)
        C[dummy + j] = C[dummy + j] + A[dummy + k] * B[k * N + j];
    }
  }
  gettimeofday(&end, NULL);

  r_time = end.tv_sec - start.tv_sec + ((double) (end.tv_usec - start.tv_usec)) / 1000000;

  return r_time;
}

double MatVecMul(double *A, double *v, double *w, size_t N) {
  struct timeval start, end;
  double r_time = 0.0;
  size_t i, j;

  size_t dummy = 0;

  ZeroVector(w, N);

  gettimeofday(&start, NULL);

  for (i = 0; i < N; ++i) {
    dummy = i * N;
    for (j = 0; j < N; ++j) {
      w[i] += A[dummy + j] * v[j];
    }
  }

  gettimeofday(&end, NULL);

  r_time = end.tv_sec - start.tv_sec + ((double) (end.tv_usec - start.tv_usec)) / 1000000;

  return r_time;
}
