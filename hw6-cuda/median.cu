
#include <stdint.h>
#include <stdio.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"


__global__ void median(uint8_t *d_img, uint8_t *d_res, size_t w, size_t h) {
    size_t x = threadIdx.x;
    size_t y = blockIdx.x;

    size_t n = 3;
    size_t k = n / 2;

    if (x < k || x >= w - k || y < k || y >= h - k) {
        d_res[y * w + x] = d_img[y * w + x];
        return;
    }

    

    unsigned char window[9] = {0,0,0,0,0,0,0,0,0};
    size_t ii, jj;
    for (size_t i = 0; i < n; ++i) {
        ii = y - k + i;
        for (size_t j = 0; j < n; ++j) {
            jj = x - k + j;
            window[i * n + j] = d_img[ii * w + jj];
        }
    }

    //sort
    for (size_t i = 0; i < 5; ++i) {
        for (size_t j = i + 1; j < 9; ++j) {
            if (window[i] > window[j]) {
                uint8_t tmp = window[i];
                window[i] = window[j];
                window[j] = tmp;
            }
        }
    }
    
    d_res[y * w + x] = window[4];
}

int main() {
    
    int width, height, channels;
    uint8_t *h_img = stbi_load("noise.png", &width, &height, &channels, 0); 

    uint8_t *d_img;
    cudaMalloc(&d_img, width * height * sizeof(uint8_t));
    cudaMemcpy(d_img, h_img, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);

    uint8_t *h_res = (uint8_t*)malloc(width * height * sizeof(uint8_t));
    uint8_t *d_res;
    cudaMalloc(&d_res, width * height * sizeof(uint8_t));

    median<<<height, width>>>(d_img, d_res, width, height);

    cudaMemcpy(h_res, d_res, width * height * sizeof(uint8_t), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();

    stbi_write_png("denoise.png", width, height, 1, h_res, width);

    free(h_res);
    cudaFree(d_res);
    cudaFree(d_img);
    stbi_image_free(h_img);
    return 0;
}