#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h>
#include <omp.h>

#define nThreads 4
unsigned int seeds[nThreads];

void seedThreads() {
  int my_thread_id;
  unsigned int seed;
#pragma omp parallel default(none) shared(seeds) private(seed, my_thread_id)
  {
    my_thread_id = omp_get_thread_num();
    seed = (unsigned) time(NULL);
    seeds[my_thread_id] = (seed & 0xFFFFFFF0) | (my_thread_id + 1);
  }
}

void print_matrix(float *m, int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      printf("%g ", m[i * n + j]);
    }
    printf("\n");
  }
  printf("\n");
}

float* binpow(float *A, int N, int p) {
  int i, j;
  float *res = (float *) malloc(sizeof(float) * N * N);
  float *res_copy = (float *) malloc(sizeof(float) * N * N);
  float *A_copy = (float *) malloc(sizeof(float) * N * N);

#pragma omp parallel for default(none) shared(N, res) private(i, j)
  for (i = 0; i < N; ++i) {
    for (j = 0; j < N; ++j) {
      if (i == j)
        res[i * N + j] = 1;
      else
        res[i * N + j] = 0;
    }
  }

  while (p) {
    if (p & 1) {
      for (i = 0; i < N * N; ++i) {
        res_copy[i] = res[i];
      }
      cblas_sgemm(
              CblasRowMajor, CblasNoTrans, CblasNoTrans,
              N, N, N, 1, res_copy, N, A, N, 0, res, N
              );
      p--;
    } else {
      for (i = 0; i < N * N; ++i) {
        A_copy[i] = A[i];
      }
      cblas_sgemm(
              CblasRowMajor, CblasNoTrans, CblasNoTrans,
              N, N, N, 1, A_copy, N, A_copy, N, 0, A, N
      );
      p >>= 1;
    }
  }

  free(res_copy);
  free(A_copy);

  return res;
}

int main(int argc, char *argv[]) {
  omp_set_num_threads(nThreads);
  seedThreads();
  int N = atoi(argv[1]);
  float *A = (float *) malloc(sizeof(float) * N * N);
  int i, j, tid;
  unsigned int seed;

#pragma omp parallel default(none) private(i, j, tid, seed) shared(seeds, N, A)
  {
    tid = omp_get_thread_num();
    seed = seeds[tid];
    srand(seed);
#pragma omp for
    for (i = 0; i < N; ++i) {
      A[i * (N + 1)] = 0;
      for (j = i + 1; j < N; ++j) {
        A[i * N + j] = A[j * N + i] = rand_r(&seed) % 2;
      }
    }
  }


  print_matrix(A, N);

  float *res = binpow(A, N, atoi(argv[2]));
  print_matrix(res, N);
  free(A);
  free(res);
  return 0;
}
