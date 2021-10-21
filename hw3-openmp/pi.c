#include <omp.h>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#define nThreads 4  // number of threads to use
unsigned int seeds[nThreads];
static long total_cnt = 1<<22;

void seedThreads() {
    int my_thread_id;
    unsigned int seed;
    #pragma omp parallel private (seed, my_thread_id)
    {
        my_thread_id = omp_get_thread_num();
        unsigned int seed = (unsigned) time(NULL);
        seeds[my_thread_id] = (seed & 0xFFFFFFF0) | (my_thread_id + 1);
    }
}


int main(int argc, char *argv[]) {
    double pi = 0;
    omp_set_num_threads(nThreads);
    seedThreads();
    
    long in_circle_cnt=0;
    int tid, i;
    double x, y;
    unsigned int seed;
    double start, end;
    start = omp_get_wtime();
#pragma omp parallel default(none) num_threads(nThreads) \
                                        shared(seeds, total_cnt) \
                                        private(tid, seed, i, x, y) \
                                        reduction(+:in_circle_cnt)
    {
        tid = omp_get_thread_num();
        seed = seeds[tid];
        srand(seed);
    
#pragma omp for
        for(i = 0; i < total_cnt; i++) {
            x = (double)rand_r(&seed) / RAND_MAX;
            y = (double)rand_r(&seed) / RAND_MAX;
            if (x*x + y*y <= 1.0) {
                in_circle_cnt += 1;
            }
        }
    }
    
    pi = 4 * (double)in_circle_cnt / total_cnt;
    
    end = omp_get_wtime();
    printf("pi = %e\n", pi);
    
    printf("Time elapsed = %f\n", (end - start));
    return 0;
}
