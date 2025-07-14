
#include <cuda_runtime.h>
#include "plugin/yolov8PoseLayerPlugin/yolov8PoseLayerParameters.h"
#include "trt/trt_cuda.h"

__device__ 
float sigmoid_gpu(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__
float iou_gpu(
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

__global__
void YOLOv8PoseLayerNMS(
    YOLOv8PoseLayerParameters param,
    float* regData, float* clsData, float* psData, int headStart, int headEnd,
    float* outputRects, int* outputClasses, int* outputCount, int* outputKeep,
    int* __restrict__ numDetectionsOutput, int* __restrict__ nmsClassesOutput, 
    float* __restrict__ nmsScoresOutput, float* __restrict__ nmsBoxesOutput, 
    float* __restrict__ nmsKeyPointsOutput
) {
    int imageIdx = blockIdx.x;
    int anchorIdx = blockIdx.y * blockDim.x + threadIdx.x;
    if (imageIdx >= param.batchSize) return;
    if (anchorIdx >= param.numAnchors) return;

    int head_idx = (anchorIdx < headStart) ? 0 : (anchorIdx < headEnd) ? 1 : 2;

    int local_start = (head_idx == 1) * headStart + (head_idx == 2) * headEnd;
    
    int stride = param.minStride << head_idx;

    int h = param.inputHeight / stride;
    int w = param.inputWidth / stride;
    int local_idx = anchorIdx - local_start;
    int row = local_idx / w;
    int col = local_idx % w;

    float cx = float(col + 0.5);
    float cy = float(row + 0.5);

    float cls_max = -1;
    int cls_index = -1;
    for (int cl = 0; cl < param.numClasses; cl++) {
        float cls_val = clsData[cl * param.numAnchors + anchorIdx];
        if (cls_val > cls_max) {
            cls_max = cls_val;
            cls_index = cl;
        }
    }

    if (cls_max < param.scoreThreshold) return;

    float dx1 = regData[0 * param.numAnchors + anchorIdx];
    float dy1 = regData[1 * param.numAnchors + anchorIdx];
    float dx2 = regData[2 * param.numAnchors + anchorIdx];
    float dy2 = regData[3 * param.numAnchors + anchorIdx];

    float xmin = (cx - dx1) * stride;
    float ymin = (cy - dy1) * stride;
    float xmax = (cx + dx2) * stride;
    float ymax = (cy + dy2) * stride;
    
    xmin = fmaxf(0.0f, xmin);
    ymin = fmaxf(0.0f, ymin);
    xmax = fminf(param.inputWidth, xmax);
    ymax = fminf(param.inputHeight, ymax);

    int batch_offset = imageIdx * param.numAnchors;

    int id = atomicAdd(&outputCount[imageIdx], 1);

    int index_i = id + batch_offset;
    outputKeep[index_i] = 1;

    float* index_i_rect = outputRects + index_i * (4 + 1);
    int* index_i_class = outputClasses + index_i;
    index_i_rect[0] = xmin;
    index_i_rect[1] = ymin;
    index_i_rect[2] = xmax;
    index_i_rect[3] = ymax;
    index_i_rect[4] = cls_max;
    index_i_class[0] = cls_index;

    __syncthreads();

    for (int j = 0; j < outputCount[imageIdx]; j++) {
        int index_j = j + batch_offset;

        float* index_j_rect = outputRects + index_j * (4 + 1);
        int* index_j_class = outputClasses + index_j;

        if (index_i == index_j || outputKeep[index_j] == 0) continue;
        if (index_i_class[0] == index_j_class[0] && index_i_rect[4] < index_j_rect[4]) {
            float iou = iou_gpu(
                index_i_rect[0], index_i_rect[1], index_i_rect[2], index_i_rect[3], 
                index_j_rect[0], index_j_rect[1], index_j_rect[2], index_j_rect[3]
            );
            if (iou > param.iouThreshold) {
                outputKeep[index_i] = 0;
            }
        }
    }

    __syncthreads();

    if (outputKeep[index_i] == 1) {
        // NumDetections
        int kid = atomicAdd(&numDetectionsOutput[imageIdx], 1);

        // DetectionClasses
        nmsClassesOutput[kid + imageIdx * param.numOutputBoxes] = index_i_class[0];

        // DetectionScores
        nmsScoresOutput[kid + imageIdx * param.numOutputBoxes] = index_i_rect[4];

        // DetectionBoxes
        nmsBoxesOutput[(kid + imageIdx * param.numOutputBoxes) * 4 + 0] = index_i_rect[0];
        nmsBoxesOutput[(kid + imageIdx * param.numOutputBoxes) * 4 + 1] = index_i_rect[1];
        nmsBoxesOutput[(kid + imageIdx * param.numOutputBoxes) * 4 + 2] = index_i_rect[2];
        nmsBoxesOutput[(kid + imageIdx * param.numOutputBoxes) * 4 + 3] = index_i_rect[3];

        // DetectionKeyPoints
        for (int k = 0; k < param.numKeypoints; k++) {
            nmsKeyPointsOutput[(k + imageIdx * param.numOutputBoxes) * param.numKeypoints * 3 + kid * 3 + 0] = (psData[(k * 3 + 0) * param.numAnchors + anchorIdx] * 2 + (cx - 0.5f)) * stride;
            nmsKeyPointsOutput[(k + imageIdx * param.numOutputBoxes) * param.numKeypoints * 3 + kid * 3 + 1] = (psData[(k * 3 + 1) * param.numAnchors + anchorIdx] * 2 + (cy - 0.5f)) * stride;
            nmsKeyPointsOutput[(k + imageIdx * param.numOutputBoxes) * param.numKeypoints * 3 + kid * 3 + 2] = sigmoid_gpu(psData[(k * 3 + 2) * param.numAnchors + anchorIdx]);
        }
    }
}

void YOLOv8PoseLayerLauncher(
    YOLOv8PoseLayerParameters param,
    float* regInput, float* clsInput, float* psInput,
    void* numDetectionsOutput, void* nmsClassesOutput, void* nmsScoresOutput, 
    void* nmsBoxesOutput, void* nmsKeyPointsOutput
) {
    checkCudaRuntime(cudaMemset(numDetectionsOutput, 0, sizeof(int) * param.batchSize));
    checkCudaRuntime(cudaMemset(nmsClassesOutput, 0, sizeof(int) * param.batchSize * param.numOutputBoxes));
    checkCudaRuntime(cudaMemset(nmsScoresOutput, 0, sizeof(float) * param.batchSize * param.numOutputBoxes));
    checkCudaRuntime(cudaMemset(nmsBoxesOutput, 0, sizeof(float) * param.batchSize * param.numOutputBoxes * 4));
    checkCudaRuntime(cudaMemset(nmsKeyPointsOutput, 0, sizeof(float) * param.batchSize * param.numOutputBoxes * param.numKeypoints * 3));

    float* outputRects = nullptr;
    int rects_element = (4 + 1) * param.batchSize * param.numAnchors;
    checkCudaRuntime(cudaMalloc(&outputRects, rects_element * sizeof(float)));

    int* outputClasses = nullptr;
    int socres_element = param.batchSize * param.numAnchors;
    checkCudaRuntime(cudaMalloc(&outputClasses, socres_element * sizeof(int)));

    int* outputCount = nullptr;
    int count_element = param.batchSize;
    checkCudaRuntime(cudaMalloc(&outputCount, count_element * sizeof(int)));
    checkCudaRuntime(cudaMemset(outputCount, 0, sizeof(int) * count_element));

    int* outputKeep = nullptr;
    int keep_element = param.batchSize * param.numAnchors;
    checkCudaRuntime(cudaMalloc(&outputKeep, keep_element * sizeof(int)));
    checkCudaRuntime(cudaMemset(outputKeep, -1, keep_element * sizeof(int)));

    int headStart = (param.inputHeight / param.minStride) * (param.inputWidth / param.minStride);
    int midMapSize = (param.inputHeight / (param.minStride << 1)) * (param.inputWidth / (param.minStride << 1));
    int headEnd = headStart + midMapSize;

    int threadSize = 256;
    dim3 block(threadSize, 1);
    dim3 grid(param.batchSize, (param.numAnchors + threadSize - 1) / threadSize);

    YOLOv8PoseLayerNMS<<<grid, block>>>(
        param,
        regInput, clsInput, psInput, headStart, headEnd,
        outputRects, outputClasses, outputCount, outputKeep,
        (int *)numDetectionsOutput, (int *)nmsClassesOutput, 
        (float *)nmsScoresOutput, (float *)nmsBoxesOutput, 
        (float *)nmsKeyPointsOutput
    );

    checkCudaRuntime(cudaFree(outputKeep));
    checkCudaRuntime(cudaFree(outputCount));
    checkCudaRuntime(cudaFree(outputClasses));
    checkCudaRuntime(cudaFree(outputRects));
}

void YOLOv8PoseLayerInference(
    YOLOv8PoseLayerParameters param,
    float* regInput, float* clsInput, float* psInput,
    int regSize, int clsSize, int psSize,
    int* numDetectionsOutput, int* nmsClassesOutput, float* nmsScoresOutput, 
    float* nmsBoxesOutput, float* nmsKeyPointsOutput
) {

    float* d_regInput = nullptr;
    float* d_clsInput = nullptr;
    float* d_psInput = nullptr;

    checkCudaRuntime(cudaMalloc(&d_regInput, regSize * sizeof(float)));
    checkCudaRuntime(cudaMalloc(&d_clsInput, clsSize * sizeof(float)));
    checkCudaRuntime(cudaMalloc(&d_psInput, psSize * sizeof(float)));

    checkCudaRuntime(cudaMemcpy(d_regInput, regInput, regSize * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaMemcpy(d_clsInput, clsInput, clsSize * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaMemcpy(d_psInput, psInput, psSize * sizeof(float), cudaMemcpyHostToDevice));
    

    void* d_numDetectionsOutput = nullptr;
    void* d_nmsClassesOutput    = nullptr;
    void* d_nmsScoresOutput     = nullptr;
    void* d_nmsBoxesOutput      = nullptr;
    void* d_nmsKeyPointsOutput  = nullptr;

    checkCudaRuntime(cudaMalloc(&d_numDetectionsOutput, sizeof(int)));
    checkCudaRuntime(cudaMalloc(&d_nmsClassesOutput, sizeof(int) * param.numOutputBoxes));
    checkCudaRuntime(cudaMalloc(&d_nmsScoresOutput, sizeof(float) * param.numOutputBoxes));
    checkCudaRuntime(cudaMalloc(&d_nmsBoxesOutput, sizeof(float) * param.numOutputBoxes * 4));
    checkCudaRuntime(cudaMalloc(&d_nmsKeyPointsOutput, sizeof(float) * param.numOutputBoxes * 3 * param.numKeypoints));

    checkCudaRuntime(cudaMemset(d_numDetectionsOutput, 0, sizeof(int)));
    checkCudaRuntime(cudaMemset(d_nmsClassesOutput, 0, sizeof(int) * param.numOutputBoxes));
    checkCudaRuntime(cudaMemset(d_nmsScoresOutput, 0, sizeof(float) * param.numOutputBoxes));
    checkCudaRuntime(cudaMemset(d_nmsBoxesOutput, 0, sizeof(float) * param.numOutputBoxes * 4));
    checkCudaRuntime(cudaMemset(d_nmsKeyPointsOutput, 0, sizeof(float) * param.numOutputBoxes * 3 * param.numKeypoints));

    YOLOv8PoseLayerLauncher(
        param,
        d_regInput, d_clsInput, d_psInput,
        d_numDetectionsOutput, d_nmsClassesOutput, d_nmsScoresOutput,
        d_nmsBoxesOutput, d_nmsKeyPointsOutput
    );

    checkCudaRuntime(cudaMemcpy(numDetectionsOutput, d_numDetectionsOutput, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaMemcpy(nmsClassesOutput, d_nmsClassesOutput, sizeof(int) * param.numOutputBoxes, cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaMemcpy(nmsScoresOutput, d_nmsScoresOutput, sizeof(float) * param.numOutputBoxes, cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaMemcpy(nmsBoxesOutput, d_nmsBoxesOutput, sizeof(float) * param.numOutputBoxes * 4, cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaMemcpy(nmsKeyPointsOutput, d_nmsKeyPointsOutput, sizeof(float) * param.numOutputBoxes * 3 * param.numKeypoints, cudaMemcpyDeviceToHost));

    checkCudaRuntime(cudaFree(d_nmsKeyPointsOutput));
    checkCudaRuntime(cudaFree(d_nmsBoxesOutput));
    checkCudaRuntime(cudaFree(d_nmsScoresOutput));
    checkCudaRuntime(cudaFree(d_nmsClassesOutput));
    checkCudaRuntime(cudaFree(d_numDetectionsOutput));

    checkCudaRuntime(cudaFree(d_psInput));
    checkCudaRuntime(cudaFree(d_clsInput));
    checkCudaRuntime(cudaFree(d_regInput));
}