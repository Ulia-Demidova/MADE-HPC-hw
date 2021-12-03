
#include <stdint.h>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

const size_t BIN = 10;

__global__ void init(int *d_hist, size_t n) {
    // printf("In init\n");
    for (size_t i = 0; i < n; ++i) {
        d_hist[threadIdx.x] = 0;
    }
}


__global__ void calc_hist(uint8_t *d_img, int *d_hist, int w) {
    //printf("In GPU\n");
    size_t tid = blockIdx.x * w + threadIdx.x;

    //printf("%c ", d_img[tid]);
    size_t idx = (int) d_img[tid] / 256.0 * BIN;
    atomicAdd(&(d_hist[idx]), 1);
}


int main() {
    
    int width, height, channels;
    uint8_t *h_img = stbi_load("lena.png", &width, &height, &channels, 0); 

    uint8_t *d_img;
    cudaMalloc(&d_img, width * height * sizeof(uint8_t));
    cudaMemcpy(d_img, h_img, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);


    int *h_hist = (int*)malloc(BIN * sizeof(int));
    int *d_hist;
    cudaMalloc(&d_hist, BIN * sizeof(int));

    init<<<1, BIN>>>(d_hist, BIN);
    calc_hist<<<height, width>>>(d_img, d_hist, width);

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