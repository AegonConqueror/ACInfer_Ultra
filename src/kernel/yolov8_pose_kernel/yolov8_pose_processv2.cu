
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
    float*  reg1Data, float*  reg2Data, float*  reg3Data, 
    float*  cls1Data, float*  cls2Data, float*  cls3Data, 
    float*  ps1Data, float*  ps2Data, float*  ps3Data,
    float* outputRects, int* outputClasses, int* outputCount, int* outputKeep,
    int* __restrict__ numDetectionsOutput, int* __restrict__ nmsClassesOutput, 
    float* __restrict__ nmsScoresOutput, float* __restrict__ nmsBoxesOutput, 
    float* __restrict__ nmsKeyPointsOutput
) {
    int imageIdx = blockIdx.x;
    int anchorIdx = blockIdx.y * blockDim.x + threadIdx.x;
    if (imageIdx >= param.batchSize) return;
    if (anchorIdx >= param.numAnchors) return;

    int head_idx = 0;
    int base_tid = anchorIdx % param.numAnchors;

    if (base_tid < param.headStart)
        head_idx = 0;
    else if (base_tid < param.headEnd)
        head_idx = 1;
    else
        head_idx = 2;

    int local_start = (head_idx == 1) * param.headStart + (head_idx == 2) * param.headEnd;
    
    int stride = param.minStride << head_idx;

    int h = param.inputHeight / stride;
    int w = param.inputWidth / stride;
    int local_idx = base_tid - local_start;
    int row = local_idx / w;
    int col = local_idx % w;
    
    float* reg = (head_idx == 0) ? reg1Data + imageIdx * param.reg1Size : ((head_idx == 1) ? reg2Data + imageIdx * param.reg2Size: reg3Data + imageIdx * param.reg3Size);
    float* cls = (head_idx == 0) ? cls1Data + imageIdx * param.cls1Size: ((head_idx == 1) ? cls2Data + imageIdx * param.cls2Size : cls3Data + imageIdx * param.cls3Size);
    float* ps  = (head_idx == 0) ? ps1Data + imageIdx * param.ps1Size : ((head_idx == 1) ? ps2Data + imageIdx * param.ps2Size : ps3Data + imageIdx * param.ps3Size);

    float cx = float(col + 0.5);
    float cy = float(row + 0.5);

    float cls_max = -1;
    int cls_index = -1;
    for (int cl = 0; cl < param.numClasses; cl++) {
        float cls_val = sigmoid_gpu(cls[cl * h * w + row * w + col]);
        if (cls_val > cls_max) {
            cls_max = cls_val;
            cls_index = cl;
        }
    }

    if (cls_max < param.scoreThreshold) return;

    float dx1 = reg[0 * h * w + row * w + col];
    float dy1 = reg[1 * h * w + row * w + col];
    float dx2 = reg[2 * h * w + row * w + col];
    float dy2 = reg[3 * h * w + row * w + col];

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
            nmsKeyPointsOutput[(k + imageIdx * param.numOutputBoxes) * param.numKeypoints * 3 + kid * 3 + 0] = (ps[(k * 3 + 0) * h * w + row * w + col] * 2 + (cx - 0.5f)) * stride;
            nmsKeyPointsOutput[(k + imageIdx * param.numOutputBoxes) * param.numKeypoints * 3 + kid * 3 + 1] = (ps[(k * 3 + 1) * h * w + row * w + col] * 2 + (cy - 0.5f)) * stride;
            nmsKeyPointsOutput[(k + imageIdx * param.numOutputBoxes) * param.numKeypoints * 3 + kid * 3 + 2] = sigmoid_gpu(ps[(k * 3 + 2) * h * w + row * w + col]);
        }
    }
}

void YOLOv8PoseLayerLauncher(
    YOLOv8PoseLayerParameters param,
    float* reg1Input, float* reg2Input, float* reg3Input,
    float* cls1Input, float* cls2Input, float* cls3Input,
    float* ps1Input, float* ps2Input, float* ps3Input,
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

    int threadSize = 256;
    dim3 block(threadSize, 1);
    dim3 grid(param.batchSize, (param.numAnchors + threadSize - 1) / threadSize);

    YOLOv8PoseLayerNMS<<<grid, block>>>(
        param,
        reg1Input, reg2Input, reg3Input, 
        cls1Input, cls2Input, cls3Input, 
        ps1Input, ps2Input, ps3Input,
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
    float* reg1Input, float* reg2Input, float* reg3Input,
    float* cls1Input, float* cls2Input, float* cls3Input,
    float* ps1Input, float* ps2Input, float* ps3Input,
    int* numDetectionsOutput, int* nmsClassesOutput, float* nmsScoresOutput, 
    float* nmsBoxesOutput, float* nmsKeyPointsOutput
) {

    float* d_reg1Input = nullptr;
    float* d_cls1Input = nullptr;
    float* d_reg2Input = nullptr;
    float* d_cls2Input = nullptr;
    float* d_reg3Input = nullptr;
    float* d_cls3Input = nullptr;
    float* d_ps1Input = nullptr;
    float* d_ps2Input = nullptr;
    float* d_ps3Input = nullptr;

    checkCudaRuntime(cudaMalloc(&d_reg1Input, param.reg1Size * sizeof(float)));
    checkCudaRuntime(cudaMalloc(&d_cls1Input, param.cls1Size * sizeof(float)));
    checkCudaRuntime(cudaMalloc(&d_reg2Input, param.reg2Size * sizeof(float)));
    checkCudaRuntime(cudaMalloc(&d_cls2Input, param.cls2Size * sizeof(float)));
    checkCudaRuntime(cudaMalloc(&d_reg3Input, param.reg3Size * sizeof(float)));
    checkCudaRuntime(cudaMalloc(&d_cls3Input, param.cls3Size * sizeof(float)));
    checkCudaRuntime(cudaMalloc(&d_ps1Input, param.ps1Size * sizeof(float)));
    checkCudaRuntime(cudaMalloc(&d_ps2Input, param.ps2Size * sizeof(float)));
    checkCudaRuntime(cudaMalloc(&d_ps3Input, param.ps3Size * sizeof(float)));

    checkCudaRuntime(cudaMemcpy(d_reg1Input, reg1Input, param.reg1Size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaMemcpy(d_cls1Input, cls1Input, param.cls1Size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaMemcpy(d_reg2Input, reg2Input, param.reg2Size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaMemcpy(d_cls2Input, cls2Input, param.cls2Size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaMemcpy(d_reg3Input, reg3Input, param.reg3Size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaMemcpy(d_cls3Input, cls3Input, param.cls3Size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaMemcpy(d_ps1Input, ps1Input, param.ps1Size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaMemcpy(d_ps2Input, ps2Input, param.ps2Size * sizeof(float), cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaMemcpy(d_ps3Input, ps3Input, param.ps3Size * sizeof(float), cudaMemcpyHostToDevice));

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
        d_reg1Input, d_reg2Input, d_reg3Input,
        d_cls1Input, d_cls2Input, d_cls3Input,
        d_ps1Input, d_ps2Input, d_ps3Input,
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

    checkCudaRuntime(cudaFree(d_ps3Input));
    checkCudaRuntime(cudaFree(d_ps2Input));
    checkCudaRuntime(cudaFree(d_ps1Input));
    checkCudaRuntime(cudaFree(d_cls3Input));
    checkCudaRuntime(cudaFree(d_reg3Input));
    checkCudaRuntime(cudaFree(d_cls2Input));
    checkCudaRuntime(cudaFree(d_reg2Input));
    checkCudaRuntime(cudaFree(d_cls1Input));
    checkCudaRuntime(cudaFree(d_reg1Input));
}