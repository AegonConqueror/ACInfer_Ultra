#include "cuda_runtime_api.h"
#include "yolov8DetLayerParameters.h"

#include <stdio.h>
#include "common.cuh"

__global__
void YOLOv8DetLayerNMS(
    YOLOv8DetLayerParameters param,
    const float* regData, const float* clsData,
    int headStart, int headEnd,
    float* outputRects, int* outputClasses, int* outputCount, int* outputKeep,
    int* __restrict__ numDetectionsOutput, int* __restrict__ nmsClassesOutput, 
    float* __restrict__ nmsScoresOutput, float* __restrict__ nmsBoxesOutput
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
    index_i_rect[4] = sigmoid(cls_max);
    index_i_class[0] = cls_index;

    __syncthreads();

    for (int j = 0; j < outputCount[imageIdx]; j++) {
        int index_j = j + batch_offset;

        float* index_j_rect = outputRects + index_j * (4 + 1);
        int* index_j_class = outputClasses + index_j;

        if (index_i == index_j || outputKeep[index_j] == 0) continue;
        if (index_i_class[0] == index_j_class[0] && index_i_rect[4] < index_j_rect[4]) {
            float iou = IoU(
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
    }
}

template <typename T>
T* YOLOv8DetLayerWorkspace(void* workspace, size_t& offset, size_t elements) {
    T* buffer = (T*) ((size_t) workspace + offset);
    size_t size = elements * sizeof(T);
    offset += size;
    return buffer;
}

pluginStatus_t YOLOv8DetLayerLauncher(
    YOLOv8DetLayerParameters param,
    const void* regInput, const void* clsInput,
    void* numDetectionsOutput, void* nmsClassesOutput, void* nmsScoresOutput, void* nmsBoxesOutput, 
    void* workspace, cudaStream_t stream
) {
    cudaMemsetAsync(numDetectionsOutput, 0, sizeof(int) * param.batchSize, stream);
    cudaMemsetAsync(nmsClassesOutput, 0, sizeof(int) * param.batchSize * param.numOutputBoxes, stream);
    cudaMemsetAsync(nmsScoresOutput, 0, sizeof(float) * param.batchSize * param.numOutputBoxes, stream);
    cudaMemsetAsync(nmsBoxesOutput, 0, sizeof(float) * param.batchSize * param.numOutputBoxes * 4, stream);

    // Counters Workspace
    size_t workspaceOffset = 0;
    int rects_element = (4 + 1) * param.batchSize * param.numAnchors;
    float* outputRects = YOLOv8DetLayerWorkspace<float>(workspace, workspaceOffset, rects_element);

    int classes_element = param.batchSize * param.numAnchors;
    int* outputClasses = YOLOv8DetLayerWorkspace<int>(workspace, workspaceOffset, classes_element);

    int count_element = param.batchSize;
    int* outputCount = YOLOv8DetLayerWorkspace<int>(workspace, workspaceOffset, count_element);
    cudaMemsetAsync(outputCount, 0, sizeof(int) * param.batchSize, stream);

    int keep_element = param.batchSize * param.numAnchors;
    int* outputKeep = YOLOv8DetLayerWorkspace<int>(workspace, workspaceOffset, keep_element);
    cudaMemsetAsync(outputKeep, -1, sizeof(int) * param.numAnchors * param.batchSize, stream);

    int headStart = (param.inputHeight / param.minStride) * (param.inputWidth / param.minStride);
    int midMapSize = (param.inputHeight / (param.minStride << 1)) * (param.inputWidth / (param.minStride << 1));
    int headEnd = headStart + midMapSize;

    int threadSize = 256;
    dim3 block(threadSize, 1);
    dim3 grid(param.batchSize, (param.numAnchors + threadSize - 1) / threadSize);

    YOLOv8DetLayerNMS<<<grid, block, 0, stream>>>(
        param,
        (const float *)regInput, (const float *)clsInput,
        headStart, headEnd,
        outputRects, outputClasses, outputCount, outputKeep,
        (int *)numDetectionsOutput, (int *)nmsClassesOutput, (float *)nmsScoresOutput, (float *)nmsBoxesOutput
    );

    cudaError_t status = cudaGetLastError();
    CSC(status, STATUS_FAILURE);

    return STATUS_SUCCESS;
}

pluginStatus_t YOLOv8DetLayerInference(
    YOLOv8DetLayerParameters param,
    const void* regInput, const void* clsInput,
    void* numDetectionsOutput, void* nmsClassesOutput, void* nmsScoresOutput, void* nmsBoxesOutput,
    void* workspace, cudaStream_t stream
) {
    return YOLOv8DetLayerLauncher(
        param,
        regInput, clsInput,
        numDetectionsOutput, nmsClassesOutput, nmsScoresOutput, nmsBoxesOutput, 
        workspace, stream 
    );
}