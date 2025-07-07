
#include "cuda_runtime_api.h"
#include "yolov8PoseLayerParameters.h"

__device__ 
float sigmoid_gpu(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__
float iou_gpu(
    const int xmin_A, const int ymin_A, const int xmax_A, const int ymax_A,
    const int xmin_B, const int ymin_B, const int xmax_B, const int ymax_B
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
    const float* reg1Data, const float* reg2Data, const float* reg3Data, 
    const float* cls1Data, const float* cls2Data, const float* cls3Data, 
    const float* ps1Data, const float* ps2Data, const float* ps3Data,
    int* outputRects, float* outputSocres, int* outputCount, int* outputKeep,
    int* __restrict__ numDetectionsOutput, int* __restrict__ nmsClassesOutput, 
    float* __restrict__ nmsScoresOutput, float* __restrict__ nmsBoxesOutput, 
    float* __restrict__ nmsKeyPointsOutput
) {
    int imageIdx = blockIdx.x;
    int anchorIdx = blockIdx.y * blockDim.x + threadIdx.x;
    if (imageIdx >= 2) return;
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
    
    const float* reg = (head_idx == 0) ? reg1Data + imageIdx * param.reg1Size : ((head_idx == 1) ? reg2Data + imageIdx * param.reg2Size: reg3Data + imageIdx * param.reg3Size);
    const float* cls = (head_idx == 0) ? cls1Data + imageIdx * param.cls1Size: ((head_idx == 1) ? cls2Data + imageIdx * param.cls2Size : cls3Data + imageIdx * param.cls3Size);
    const float* ps  = (head_idx == 0) ? ps1Data + imageIdx * param.ps1Size : ((head_idx == 1) ? ps2Data + imageIdx * param.ps2Size : ps3Data + imageIdx * param.ps3Size);

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

    int* index_i_rect = outputRects + index_i * (4 + 1);
    float* index_i_score = outputSocres + index_i;
    index_i_rect[0] = int(xmin / param.inputWidth + 0.5);
    index_i_rect[1] = int(ymin / param.inputHeight + 0.5);
    index_i_rect[2] = int(xmax / param.inputWidth + 0.5);
    index_i_rect[3] = int(ymax / param.inputHeight + 0.5);
    index_i_rect[4] = cls_index;
    index_i_score[0] = cls_max;

    __syncthreads();

    for (int j = 0; j < outputCount[imageIdx]; j++) {
        int index_j = j + batch_offset;

        int* index_j_rect = outputRects + index_j * (4 + 1);
        float* index_j_score = outputSocres + index_j;

        if (index_i == index_j || outputKeep[index_j] == 0) continue;
        if (index_i_score[0] < index_j_score[0] &&  index_i_rect[4] == index_j_rect[4]) {
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
        nmsClassesOutput[kid + imageIdx * param.numOutputBoxes] = index_i_rect[4];

        // DetectionScores
        nmsScoresOutput[kid + imageIdx * param.numOutputBoxes] = index_i_score[0];

        // DetectionBoxes
        nmsBoxesOutput[(kid + imageIdx * param.numOutputBoxes) * 4 + 0] = index_i_rect[0];
        nmsBoxesOutput[(kid + imageIdx * param.numOutputBoxes) * 4 + 1] = index_i_rect[1];
        nmsBoxesOutput[(kid + imageIdx * param.numOutputBoxes) * 4 + 2] = index_i_rect[2];
        nmsBoxesOutput[(kid + imageIdx * param.numOutputBoxes) * 4 + 3] = index_i_rect[3];

        // DetectionKeyPoints
        // float* det_keypoints_batch = det_keypoints + imageIdx * topK * 3 * num_keypoints;
        for (int k = 0; k < param.numKeypoints; k++) {
            nmsKeyPointsOutput[(k + imageIdx * param.numOutputBoxes) * param.numKeypoints * 3 + kid * 3 + 0] = (ps[(k * 3 + 0) * h * w + row * w + col] * 2 + (cx - 0.5f)) * stride  / param.inputWidth;
            nmsKeyPointsOutput[(k + imageIdx * param.numOutputBoxes) * param.numKeypoints * 3 + kid * 3 + 1] = (ps[(k * 3 + 1) * h * w + row * w + col] * 2 + (cy - 0.5f)) * stride  / param.inputHeight;
            nmsKeyPointsOutput[(k + imageIdx * param.numOutputBoxes) * param.numKeypoints * 3 + kid * 3 + 2] = sigmoid_gpu(ps[(k * 3 + 2) * h * w + row * w + col]);
        }
    }
}

template <typename T>
T* YOLOv8PoseLayerWorkspace(void* workspace, size_t& offset, size_t elements) {
    T* buffer = (T*) ((size_t) workspace + offset);
    size_t size = elements * sizeof(T);
    offset += size;
    return buffer;
}

pluginStatus_t YOLOv8PoseLayerLauncher(
    YOLOv8PoseLayerParameters param,
    const void* reg1Input, const void* reg2Input, const void* reg3Input,
    const void* cls1Input, const void* cls2Input, const void* cls3Input,
    const void* ps1Input, const void* ps2Input, const void* ps3Input,
    void* numDetectionsOutput, void* nmsClassesOutput, void* nmsScoresOutput, 
    void* nmsBoxesOutput, void* nmsKeyPointsOutput, void* workspace, cudaStream_t stream
) {
    cudaMemsetAsync(numDetectionsOutput, 0, sizeof(int) * param.batchSize, stream);
    cudaMemsetAsync(nmsClassesOutput, 0, sizeof(int) * param.batchSize * param.numOutputBoxes, stream);
    cudaMemsetAsync(nmsScoresOutput, 0, sizeof(float) * param.batchSize * param.numOutputBoxes, stream);
    cudaMemsetAsync(nmsBoxesOutput, 0, sizeof(float) * param.batchSize * param.numOutputBoxes * 4, stream);
    cudaMemsetAsync(nmsKeyPointsOutput, 0, sizeof(float) * param.batchSize * param.numOutputBoxes * param.numKeypoints * 3, stream);

    // Counters Workspace
    size_t workspaceOffset = 0;
    int rects_element = (4 + 1) * param.batchSize * param.numAnchors;
    int* outputRects = YOLOv8PoseLayerWorkspace<int>(workspace, workspaceOffset, rects_element);

    int socres_element = param.batchSize * param.numAnchors;
    float* outputScores = YOLOv8PoseLayerWorkspace<float>(workspace, workspaceOffset, socres_element);

    int count_element = param.batchSize;
    int* outputCount = YOLOv8PoseLayerWorkspace<int>(workspace, workspaceOffset, count_element);
    cudaMemsetAsync(outputCount, 0, sizeof(int) * param.batchSize, stream);

    int keep_element = param.batchSize * param.numAnchors;
    int* outputKeep = YOLOv8PoseLayerWorkspace<int>(workspace, workspaceOffset, keep_element);
    cudaMemsetAsync(outputKeep, -1, sizeof(int) * param.numAnchors * param.batchSize, stream);

    int threadSize = 256;
    dim3 block(threadSize, 1);
    dim3 grid(param.batchSize, (param.numAnchors + threadSize - 1) / threadSize);

    YOLOv8PoseLayerNMS(
        param,
        (const float *)reg1Input, (const float *)reg2Input, (const float *)reg3Input, 
        (const float *)cls1Input, (const float *)cls2Input, (const float *)cls3Input, 
        (const float *)ps1Input, (const float *)ps2Input, (const float *)ps3Input,
        outputRects, outputScores, outputCount, outputKeep,
        (int *)numDetectionsOutput, (int *)nmsClassesOutput, 
        (float *)nmsScoresOutput, (float *)nmsBoxesOutput, 
        (float *)nmsKeyPointsOutput
    );

    cudaError_t status = cudaGetLastError();
    CSC(status, STATUS_FAILURE);

    return STATUS_SUCCESS;
}

pluginStatus_t YOLOv8PoseLayerInference(
    YOLOv8PoseLayerParameters param,
    const void* reg1Input, const void* reg2Input, const void* reg3Input,
    const void* cls1Input, const void* cls2Input, const void* cls3Input,
    const void* ps1Input, const void* ps2Input, const void* ps3Input,
    void* numDetectionsOutput, void* nmsClassesOutput, void* nmsScoresOutput, 
    void* nmsBoxesOutput, void* nmsKeyPointsOutput, void* workspace, cudaStream_t stream
) {
    return YOLOv8PoseLayerLauncher(
        param,
        reg1Input, reg2Input, reg3Input,
        cls1Input, cls2Input, cls3Input,
        ps1Input, ps2Input, ps3Input,
        numDetectionsOutput, nmsClassesOutput, nmsScoresOutput,
        nmsBoxesOutput, nmsKeyPointsOutput, workspace, stream 
    );
}