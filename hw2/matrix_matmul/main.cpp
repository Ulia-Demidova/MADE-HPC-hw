#include <vector>
#include "matmul.h"

int main() {
  double *A, *B, *C, *v, *w;
  double runtimeMat, runtimeVec, average_runtime_mat, average_runtime_vec;
  size_t NRuns = 5;
  std::vector<size_t> size = {500, 512, 1000, 1024, 2000, 2048};

  for (size_t n : size) {
    A = new double[n * n];
    B = new double[n * n];
    C = new double[n * n];
    v = new double[n];
    w = new double[n];

    RandomMatrix(A, n);
    RandomMatrix(B, n);
    RandomVector(v, n);

    average_runtime_mat = 0.0;
    average_runtime_vec = 0.0;

    printf("N = %zu\n", n);

    for(size_t i = 0; i < NRuns; i++) {
      runtimeMat = MatMatMul(A, B, C, n);
      printf("Matrix-Matrix runtime %lf seconds\n", runtimeMat);
      runtimeVec = MatVecMul(A, v, w, n);
      printf("Matrix-Vector runtime %lf seconds\n", runtimeVec);
      average_runtime_mat += runtimeMat;
      average_runtime_vec += runtimeVec;
    }
    average_runtime_mat /= NRuns;
    average_runtime_vec /= NRuns;
    printf("Matrix-Matrix average runtime %lf seconds\n", average_runtime_mat);
    printf("Matrix-Vector average runtime %lf seconds\n", average_runtime_vec);
    printf("---------------------------------\n");

    delete[] A;
    delete[] B;
    delete[] C;
    delete v;
    delete w;
  }
  return 0;
}
