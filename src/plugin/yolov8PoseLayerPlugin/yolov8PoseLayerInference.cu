
#include <stdint.h>
#include "trt/trt_cuda.h"

struct KeyPointGPU {
    float x, y;
    float score;
};

struct PoseRectGPU {
    float xmin, ymin, xmax, ymax;
    int classId;
    float score;
    KeyPointGPU keypoints[17];
};

__device__ 
float sigmoid_gpu(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ 
float sigmoid_gpu(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__
float iou_gpu(const PoseRectGPU &a, const PoseRectGPU &b) {
    float xmin = fmaxf(a.xmin, b.xmin);
    float ymin = fmaxf(a.ymin, b.ymin);
    float xmax = fminf(a.xmax, b.xmax);
    float ymax = fminf(a.ymax, b.ymax);
    float iw = fmaxf(0.0f, xmax - xmin);
    float ih = fmaxf(0.0f, ymax - ymin);
    float inter = iw * ih;
    float area1 = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    float area2 = (b.xmax - b.xmin) * (b.ymax - b.ymin);
    return inter / (area1 + area2 - inter);
}

__global__
void nms_kernel(const PoseRectGPU* dets, int* keep, int num, float iou_thresh) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num || keep[i] == 0) return;
    for (int j = 0; j < num; j++) {
        if (i == j || keep[j] == 0) continue;
        if (dets[i].score < dets[j].score && dets[i].classId == dets[j].classId) {
            float iou = iou_gpu(dets[i], dets[j]);
            if (iou > iou_thresh) keep[i] = 0;
        }
    }
}

__global__
void yolov8_pose_decode_kernel(
    float* input, int* obj_count, PoseRectGPU* outputRects,
    const uint& maxStride, const uint& numClasses, const uint& keyPoints, const uint& totalAnchors,
    const uint& input_w, const uint& input_h,
    const float& scoreThreshold,
    const uint* mapSize, const uint* headStarts
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= totalAnchors) return;

    int head_idx = -1;
    for (int i = 0; i < 3; i++) {
        int start = headStarts[i];
        int end = (i + 1 < 3) ? headStarts[i + 1] : totalAnchors;
        if (tid >= start && tid < end) {
            head_idx = i;
            break;
        }
    }

    if (head_idx == -1) return;

    int h = mapSize[head_idx * 2 ];
    int w = mapSize[head_idx * 2];
    int local_idx = tid - headStarts[head_idx];
    int row = local_idx / w;
    int col = local_idx % w;

    int stride_index = static_cast<int>(0.5 * head_idx * head_idx - 2.5 * head_idx + 4);
    int stride = maxStride / stride_index;

    int reg_input_size = 4 * mapSize[0] * mapSize[0];
    int cls_input_size = numClasses * mapSize[0] * mapSize[0];
    int pose_input_size = 3 * keyPoints * mapSize[0] * mapSize[0];

    float* reg = input + head_idx * (reg_input_size + cls_input_size);
    float* cls = reg + reg_input_size;
    float* ps  = input + 3 * (reg_input_size + cls_input_size) + head_idx * pose_input_size;
    float* meshgrid = input + 3 * (reg_input_size + cls_input_size + pose_input_size);

    float cx = meshgrid[tid * 2];
    float cy = meshgrid[tid * 2 + 1];

    float cls_max = -1;
    int cls_index = -1;
    for (int cl = 0; cl < numClasses; cl++) {
        float cls_val = sigmoid_gpu(cls[cl * h * w + row * w + col]);
        if (cls_val > cls_max) {
            cls_max = cls_val;
            cls_index = cl;
        }
    }

    if (cls_max < scoreThreshold) return;

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
    xmax = fminf(input_w, xmax);
    ymax = fminf(input_h, ymax);

    int id = atomicAdd(obj_count, 1);

    PoseRectGPU& temp = outputRects[id];
    temp.xmin = xmin / input_w;
    temp.ymin = ymin / input_h;
    temp.xmax = xmax / input_w;
    temp.ymax = ymax / input_h;
    temp.classId = cls_index;
    temp.score = cls_max;

    for (int kc = 0; kc < keyPoints; kc++) {
        temp.keypoints[kc].x = (ps[(kc * 3 + 0) * h * w + row * w + col] * 2 + (cx - 0.5f)) * stride / input_w;
        temp.keypoints[kc].y = (ps[(kc * 3 + 1) * h * w + row * w + col] * 2 + (cy - 0.5f)) * stride / input_h;
        temp.keypoints[kc].score = sigmoid_gpu(ps[(kc * 3 + 2) * h * w + row * w + col]);
    }
}


cudaError_t yolov8PoseLayerInfernece(
    float* input, int* num_dets, int* det_classes, float* det_scores, float* det_boxes, float* det_keypoints,
    const uint& batchSize, uint64_t& inputSize, uint64_t& outputSize, const uint& maxStride, 
    const uint& numClasses, const uint& keyPoints, const uint& totalAnchors, 
    const uint* mapSize, const uint* headStarts,
    const float& scoreThreshold, const float& nmsThreshold,
    cudaStream_t stream
) {
    int reg1_input_size = 4 * mapSize[0] * mapSize[0];
    int cls1_input_size = numClasses * mapSize[0] * mapSize[0];
    int ps1_input_size = 3 * keyPoints * mapSize[0] * mapSize[0];

    int threads = 256;
    int blocks = (outputSize + threads - 1) / threads;

    uint input_w = mapSize[2] * maxStride;
    uint input_h = mapSize[2] * maxStride;

    PoseRectGPU* d_output_objects;
    checkCudaRuntime(cudaMalloc(&d_output_objects, sizeof(PoseRectGPU) * totalAnchors));
    int* d_objectCount;
    checkCudaRuntime(cudaMalloc(&d_objectCount, sizeof(int)));
    checkCudaRuntime(cudaMemset(d_objectCount, 0, sizeof(int)));

    for (unsigned int batch = 0; batch < batchSize; ++batch) {
        yolov8_pose_decode_kernel<<<blocks, threads, 0, stream>>>(
            reinterpret_cast<float*>(input) + (batch * inputSize), d_objectCount, d_output_objects, 
            maxStride, numClasses, keyPoints, totalAnchors, input_w, input_h, scoreThreshold, mapSize, headStarts
        );

        int object_num;
        checkCudaRuntime(cudaMemcpy(&object_num, d_objectCount, sizeof(int), cudaMemcpyDeviceToHost));

        int* d_keep;
        checkCudaRuntime(cudaMalloc(&d_keep, sizeof(int) * object_num));
        checkCudaRuntime(cudaMemset(d_keep, 1, sizeof(int) * object_num));
        
        checkCudaKernel(
            nms_kernel<<<blocks, threads, 0, stream>>>(d_output_objects, d_keep, object_num, nmsThreshold);
        );

        std::vector<PoseRectGPU> h_objects(object_num);
        std::vector<int> h_keep(object_num);
        checkCudaRuntime(cudaMemcpy(h_objects.data(), d_output_objects, sizeof(PoseRectGPU) * object_num, cudaMemcpyDeviceToHost));
        checkCudaRuntime(cudaMemcpy(h_keep.data(), d_keep, sizeof(int) * object_num, cudaMemcpyDeviceToHost));

        // NumDetections [batch_size, 1] 
        num_dets[batch] = object_num;

        for (int i = 0; i < object_num; i++) {
            if (h_keep[i] == 0) continue;
            auto& obj = h_objects[i];

            // DetectionClasses [batch_size, numboxes]
            det_classes[batch * outputSize + i] = obj.classId;

            // DetectionScores [batch_size, numboxes]
            det_scores[batch * outputSize + i] = obj.score;

            // DetectionBoxes [batch_size, numboxes, 4]
            det_boxes[batch * outputSize * 4 + i * 4 + 0] = obj.xmin;
            det_boxes[batch * outputSize * 4 + i * 4 + 1] = obj.ymin;
            det_boxes[batch * outputSize * 4 + i * 4 + 2] = obj.xmax;
            det_boxes[batch * outputSize * 4 + i * 4 + 3] = obj.ymax;

            // DetectionKeyPoints [batch_size, numboxes, 3]
            for (int k = 0; k < keyPoints; k++) {
                det_keypoints[batch * outputSize * 3 + k * 3 + i + 0] = obj.keypoints[k].x;
                det_keypoints[batch * outputSize * 3 + k * 3 + i + 1] = obj.keypoints[k].y;
                det_keypoints[batch * outputSize * 3 + k * 3 + i + 2] = obj.keypoints[k].score;
            }
        }
    }
    
    return cudaGetLastError();
}