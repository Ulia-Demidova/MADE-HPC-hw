#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"


__global__ void init(int *d_hist, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        d_hist[threadIdx.x] = 0;
    }
}


__global__ void calc_hist(uint8_t *d_img, int *d_hist, int w, int bins) {
    size_t tid = blockIdx.x * w + threadIdx.x;

    size_t idx = (int) d_img[tid] / 256.0 * bins;
    atomicAdd(&(d_hist[idx]), 1);
}


int main(int argc, char **argv) {
    
    int width, height, channels;
    uint8_t *h_img = stbi_load("lena.png", &width, &height, &channels, 0); 

    uint8_t *d_img;
    cudaMalloc(&d_img, width * height * sizeof(uint8_t));
    cudaMemcpy(d_img, h_img, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);


    int *h_hist = (int*)malloc(BIN * sizeof(int));
    int *d_hist;
    cudaMalloc(&d_hist, BIN * sizeof(int));

    int BIN = atoi(argv[1]);

    init<<<1, BIN>>>(d_hist, BIN);
    calc_hist<<<height, width>>>(d_img, d_hist, width, BIN);

    cudaMemcpy(h_hist, d_hist, BIN * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();


    for (size_t i = 0; i < BIN; ++i) {
       printf("%d ", h_hist[i]);
    }
    printf("\n");

    free(h_hist);
    cudaFree(d_hist);
    cudaFree(d_img);
    stbi_image_free(h_img);
    return 0;
}