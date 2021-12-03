## Convolution

To work with images: \
`git clone https://github.com/nothings/stb.git`

Compilation \
`nvcc conv.cu` 

Convolution with 5x5 mean filter:\
`./a.out mean` \
Convolution with 3x3 laplace filter:\
`./a.out laplace`

The output image will be stored in *lena_filtered.png* file.

## Median filter

Use `median.cu`. The input image is *noise.png*, the output will be stored in *denoise.png* file.

## Histogram

Use `historgam.cu` with desired number of bins as command argument.
