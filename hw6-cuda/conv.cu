
#include <stdio.h>
#include <stdint.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"


__global__ void init_mean_filter(float *d_filter) {
    size_t tid = threadIdx.x;
    d_filter[tid] = 1.0 / 25;
}

__global__ void init_laplace_filter(float *d_filter) {
    size_t tid = threadIdx.x;
    if (tid == 1 || tid == 3 || tid == 5 || tid == 7) {
        d_filter[tid] = 1.0;
    } else if (tid == 4) {
        d_filter[tid] = -4.0;
    } else {
        d_filter[tid] = 0.0;
    }
}


__global__ void conv(uint8_t *d_img, uint8_t *d_res, size_t w, size_t h,
                     float *d_filter, size_t n) {
                         
    extern __shared__ float sdata[];
    size_t tid = threadIdx.x;

    if (tid < n * n) {
        sdata[tid] = d_filter[tid];
    }

    __syncthreads();

    size_t x = tid;
    size_t y = blockIdx.x;
    
    float res = 0.0;
    size_t ii, jj;

    for (size_t i = 0; i < n; ++i) {
        ii = y - n / 2 + i;
        if (ii < 0 || ii >= h) {
            continue;
        }
        for (size_t j = 0; j < n; ++j) {
            jj = x - n / 2 + j;
            if (jj < 0 || jj >= w) {
                continue;
            }
            res += sdata[i * n + j] * d_img[ii * w + jj];
        }
    }

    d_res[y * w + x] = (uint8_t)res;
}


int main(int argc, char **argv) {
    int width, height, channels;
    uint8_t *h_img = stbi_load("lena.png", &width, &height, &channels, 0); 
    

    uint8_t *d_img;
    uint8_t *d_res;
    float *d_filter;

    cudaMalloc(&d_img, width * height * sizeof(uint8_t));
    uint8_t *h_res = (uint8_t*)malloc(width * height * sizeof(uint8_t));
    cudaMemcpy(d_img, h_img, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

    cudaMalloc(&d_res, width * height * sizeof(uint8_t));
    
    size_t n = 0;

    if (!strcmp(argv[1], "mean")) {
        n = 5;
        cudaMalloc(&d_filter, 25 * sizeof(float));
        init_mean_filter<<<1, 25>>>(d_filter);
    } 
    else if (!strcmp(argv[1], "laplace")) {
        n = 3;
        cudaMalloc(&d_filter, 9 * sizeof(float));
        init_laplace_filter<<<1, 9>>>(d_filter);
    }

    conv<<<height, width, n * n * sizeof(float)>>>(d_img, d_res, width, height, d_filter, n);
    cudaMemcpy(h_res, d_res, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaFree(d_img);
    cudaFree(d_res);
    cudaFree(d_filter);

    stbi_write_png("lena_filtered.png", width, height, 1, h_res, width);

    free(h_res);

    stbi_image_free(h_img);

    return 0;
}