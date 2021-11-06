#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

#define N 194
#define epsilon 1e-5

float *pagerank(float *A, int n) {
  float *x = (float*) malloc(sizeof(float) * n);
  for (int i = 0; i < n; ++i)
    x[i] = (float)rand();

  float sum = cblas_sasum(n, x, 1);
  cblas_sscal(n, 1 / sum, x, 1);

  float *x_next = (float*) malloc(sizeof(float) * n);
  float *add_part = (float*) malloc(sizeof(float) * n);
  for (int i = 0; i < n; ++i)
    add_part[i] = 1.f / n;

  float alpha = 0.9f;
  float norm;

  while(1) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans, n, n, alpha, A, n, x, 1, 0, x_next, 1);
    cblas_saxpy(n, (1 - alpha), add_part, 1, x_next, 1);
    sum = cblas_sasum(n, x_next, 1);
    cblas_sscal(n, 1 / sum, x_next, 1);
    cblas_saxpy(n, -1, x_next, 1, x, 1);
    norm = cblas_snrm2(n, x, 1);
    cblas_scopy(n, x_next, 1, x, 1);
    if (norm < epsilon)
      break;
  }
  free(x_next);
  free(add_part);
  return x;
}

float *naive_ranking(float *A, int n) {
  float *x = (float*) malloc(sizeof(float) * n);
  float *res = (float*) malloc(sizeof(float) * n);
  for (int i = 0; i < n; ++i)
    x[i] = 1.f;
  cblas_sgemv(CblasRowMajor, CblasNoTrans, n, n, 1, A, n, x, 1, 0, res, 1);
  float sum = cblas_sasum(n, res, 1);
  cblas_sscal(n, 1 / sum, res, 1);
  free(x);
  return res;
}

int main() {
  FILE *file = fopen("/Users/yuliya/MADE/HPC/hw4/metro_matrix.txt", "r");
  float *A = (float*) malloc(sizeof(float) * N * N);
  int i=0;
  int edge;
  do {
    fscanf(file, "%d", &edge);
    A[i] = edge;
    i++;
  } while (!feof(file));
  fclose(file);

  float *v = pagerank(A, N);
  float *naive = naive_ranking(A, N);
  int idx = cblas_isamax(N, v, 1);
  int idx_naive = cblas_isamax(N, naive, 1);
  printf("Pagerank most popular station id: %d\n", idx);
  printf("Naive most popular station id: %d\n", idx_naive);
  free(A);
  free(v);
  free(naive);
  return 0;
}

