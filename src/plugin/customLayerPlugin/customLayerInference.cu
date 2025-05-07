

// output = (input + offset) * mul * alpha
__global__ void custom_layer(float *input, float *offset, float *mul, float *alpha, float *output) {
    const int idx = threadIdx.x;
    output[idx] = (input[idx] + offset[idx]) * mul[idx] * alpha[0];
}

cudaError_t custom_layer_inference(float *input, float *offset, float *mul, float *alpha, float *output);

cudaError_t custom_layer_inference(float *input, float *offset, float *mul, float *alpha, float *output) {
    custom_layer<<<1, 24>>>(input, offset, mul, alpha, output);
    return cudaGetLastError();
}