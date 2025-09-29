#ifndef ACINFER_ULTRA_COMMON_CUH
#define ACINFER_ULTRA_COMMON_CUH

__device__ 
inline float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ 
inline float unsigmoid(float y) {
    return -1.0 * logf((1.0 / y) - 1.0);
}

__device__
inline float IoU(
    const float xmin_A, const float ymin_A, const float xmax_A, const float ymax_A,
    const float xmin_B, const float ymin_B, const float xmax_B, const float ymax_B
) {
    float xmin = fmaxf(xmin_A, xmin_B);
    float ymin = fmaxf(ymin_A, ymin_B);
    float xmax = fminf(xmax_A, xmax_B);
    float ymax = fminf(ymax_A, ymax_B);
    float iw = fmaxf(0.0f, xmax - xmin);
    float ih = fmaxf(0.0f, ymax - ymin);
    float inter = iw * ih;
    float area1 = (xmax_A- xmin_A) * (ymax_A - ymin_A);
    float area2 = (xmax_B - xmin_B) * (ymax_B - ymin_B);
    return inter / (area1 + area2 - inter);
}

#endif // ACINFER_ULTRA_COMMON_CUH