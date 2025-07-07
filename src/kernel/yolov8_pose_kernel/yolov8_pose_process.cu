#include <cuda_runtime.h>

#include "trt/trt_cuda.h"

struct KeyPointGPU {
    float x, y;
    float score;
};

struct PoseRectGPU {
    int imageId;
    int xmin, ymin, xmax, ymax;
    int classId;
    float score;
    KeyPointGPU keypoints[17];
};

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
void yolov8_pose_decode_kernel(
    float* pBlob, int* pBlob_offset, PoseRectGPU* outputRects, int* obj_count, int* keep,
    int* objectNum, int* det_classes, float* det_scores, float* det_boxes, float* det_keypoints,
    int total_size, int headNum, int total_anchors, int head_start, int head_end, int min_stride, 
    int input_w, int input_h, int image_w, int image_h, 
    int num_class, int num_keypoints, float conf_thresh, float nms_thresh
){
    int imageIdx = blockIdx.x;
    int tid = blockIdx.y * blockDim.x + threadIdx.x;
    if (imageIdx >= 2) return;
    if (tid >= total_anchors) return;

    int head_idx = 0;
    int base_tid = tid % total_anchors;
    
    if (base_tid < head_start)
        head_idx = 0;
    else if (base_tid < head_end)
        head_idx = 1;
    else
        head_idx = 2;
    
    int local_start = (head_idx == 1) * head_start + (head_idx == 2) * head_end;

    int stride = min_stride << head_idx;
    
    int h = input_h / stride;
    int w = input_w / stride;
    int local_idx = base_tid - local_start;
    int row = local_idx / w;
    int col = local_idx % w;

    int batch_offset = imageIdx * total_anchors;
    
    float* reg = pBlob + pBlob_offset[head_idx * 2 + 0] + imageIdx * total_size;
    float* cls = pBlob + pBlob_offset[head_idx * 2 + 1] + imageIdx * total_size;
    float* ps  = pBlob + pBlob_offset[head_idx + headNum * 2] + imageIdx * total_size;
    
    float cx = float(col + 0.5);
    float cy = float(row + 0.5);

    float cls_max = -1;
    int cls_index = -1;
    for (int cl = 0; cl < num_class; cl++) {
        float cls_val = sigmoid_gpu(cls[cl * h * w + row * w + col]);
        if (cls_val > cls_max) {
            cls_max = cls_val;
            cls_index = cl;
        }
    }

    if (cls_max < conf_thresh) return;

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

    int id = atomicAdd(&obj_count[imageIdx], 1);
    keep[id + batch_offset] = 1;

    PoseRectGPU& temp = outputRects[id + batch_offset];
    temp.imageId = imageIdx;
    temp.xmin = int(xmin * image_w / input_w + 0.5);
    temp.ymin = int(ymin * image_h / input_h + 0.5);
    temp.xmax = int(xmax * image_w / input_w + 0.5);
    temp.ymax = int(ymax * image_h / input_h + 0.5);
    temp.classId = cls_index;
    temp.score = cls_max;

    for (int kc = 0; kc < num_keypoints; kc++) {
        temp.keypoints[kc].x = (ps[(kc * 3 + 0) * h * w + row * w + col] * 2 + (cx - 0.5f)) * stride * image_w / input_w;
        temp.keypoints[kc].y = (ps[(kc * 3 + 1) * h * w + row * w + col] * 2 + (cy - 0.5f)) * stride * image_h / input_h;
        temp.keypoints[kc].score = sigmoid_gpu(ps[(kc * 3 + 2) * h * w + row * w + col]);
    }

    __syncthreads();

    for (int j = 0; j < obj_count[imageIdx]; j++) {
        if (id + batch_offset == j + batch_offset || keep[j + batch_offset] == 0) continue;
        if (
            outputRects[id + batch_offset].score < outputRects[j + batch_offset].score && 
            outputRects[id + batch_offset].classId == outputRects[j + batch_offset].classId &&
            outputRects[id + batch_offset].imageId == outputRects[j + batch_offset].imageId
        ) {
            float iou = iou_gpu(outputRects[id + batch_offset], outputRects[j + batch_offset]);
            if (iou > nms_thresh) {
                keep[id + batch_offset] = 0;
            }
        }
    }

    __syncthreads();

    if (keep[id + batch_offset] == 1) {
        // NumDetections
        int kid = atomicAdd(&objectNum[imageIdx], 1);

        // DetectionClasses
        det_classes[kid + imageIdx * 20] = outputRects[id + batch_offset].classId;

        // DetectionScores
        det_scores[kid + imageIdx * 20] = outputRects[id + batch_offset].score;

        // DetectionBoxes
        det_boxes[(kid + imageIdx * 20) * 4 + 0] = outputRects[id + batch_offset].xmin;
        det_boxes[(kid + imageIdx * 20) * 4 + 1] = outputRects[id + batch_offset].ymin;
        det_boxes[(kid + imageIdx * 20) * 4 + 2] = outputRects[id + batch_offset].xmax;
        det_boxes[(kid + imageIdx * 20) * 4 + 3] = outputRects[id + batch_offset].ymax;

        // DetectionKeyPoints
        // float* det_keypoints_batch = det_keypoints + imageIdx * 20 * 3 * num_keypoints;
        for (int k = 0; k < num_keypoints; k++) {
            det_keypoints[(k + imageIdx * 20) * num_keypoints * 3 + kid * 3 + 0] = outputRects[id + batch_offset].keypoints[k].x;
            det_keypoints[(k + imageIdx * 20) * num_keypoints * 3 + kid * 3 + 1] = outputRects[id + batch_offset].keypoints[k].y;
            det_keypoints[(k + imageIdx * 20) * num_keypoints * 3 + kid * 3 + 2] = outputRects[id + batch_offset].keypoints[k].score;
        }
    }
}

void yolov8_pose_decode_gpu(
    float* output_data, int* output_size, int output_num,
    int* num_dets, int* det_classes, float* det_scores, float* det_boxes, float* det_keypoints,
    int input_w, int input_h, int image_w, int image_h, int class_num,
    float conf_thresh, float nms_thresh, int keypoint_num
) {

    int headNum = 3;
    int min_stride = 8;
    int total_anchors = 0;
    int head_start[headNum];
    for (int i = 0; i < headNum; ++i){
        int stride = min_stride << i;
        int total = static_cast<int>((input_w * input_h) / (stride * stride));
        head_start[i] = total_anchors;
        total_anchors += total;
    }

    int total_size = 0;
    int pBlob_offset[output_num];
    for (size_t i = 0; i < output_num; i++){
        pBlob_offset[i] = total_size;
        total_size += output_size[i];
    }

    float* d_output_data = nullptr;
    checkCudaRuntime(cudaMalloc(&d_output_data, total_size * sizeof(float)));
    checkCudaRuntime(cudaMemcpy(d_output_data, output_data, total_size * sizeof(float), cudaMemcpyHostToDevice));
    
    int* d_pBlob_offset = nullptr;
    checkCudaRuntime(cudaMalloc(&d_pBlob_offset, output_num * sizeof(int)));
    checkCudaRuntime(cudaMemcpy(d_pBlob_offset, pBlob_offset, output_num * sizeof(int), cudaMemcpyHostToDevice));
    
    PoseRectGPU* d_output_objects;
    int* d_objectCount;
    int* d_keep;
    checkCudaRuntime(cudaMalloc(&d_output_objects, sizeof(PoseRectGPU) * total_anchors));
    checkCudaRuntime(cudaMalloc(&d_objectCount, sizeof(int)));
    checkCudaRuntime(cudaMemset(d_objectCount, 0, sizeof(int)));
    checkCudaRuntime(cudaMalloc(&d_keep, sizeof(int) * total_anchors));
    checkCudaRuntime(cudaMemset(d_keep, -1, sizeof(int) * total_anchors));

    int keepTopK = 20;

    int* num_detections;
    int* detection_classes;
    float* detection_scores;
    float* detection_boxes;
    float* detection_keypoints;

    checkCudaRuntime(cudaMalloc(&num_detections, sizeof(int)));
    checkCudaRuntime(cudaMalloc(&detection_classes, sizeof(int) * keepTopK));
    checkCudaRuntime(cudaMalloc(&detection_scores, sizeof(float) * keepTopK));
    checkCudaRuntime(cudaMalloc(&detection_boxes, sizeof(float) * keepTopK * 4));
    checkCudaRuntime(cudaMalloc(&detection_keypoints, sizeof(float) * keepTopK * 3 * keypoint_num));

    checkCudaRuntime(cudaMemset(num_detections, 0, sizeof(int)));
    checkCudaRuntime(cudaMemset(detection_classes, 0, sizeof(int) * keepTopK));
    checkCudaRuntime(cudaMemset(detection_scores, 0, sizeof(float) * keepTopK));
    checkCudaRuntime(cudaMemset(detection_boxes, 0, sizeof(float) * keepTopK * 4));
    checkCudaRuntime(cudaMemset(detection_keypoints, 0, sizeof(float) * keepTopK * 3 * keypoint_num));

    int threadSize = 256;

    dim3 block(threadSize, 1);
    dim3 grid(1, (total_anchors + threadSize - 1) / threadSize);

    checkCudaKernel(
        yolov8_pose_decode_kernel<<<grid, block>>>(
            d_output_data, d_pBlob_offset, d_output_objects, d_objectCount, d_keep, 
            num_detections, detection_classes, detection_scores, detection_boxes, detection_keypoints,
            total_size, headNum, total_anchors, head_start[1], head_start[2], min_stride, 
            input_w, input_h, image_w, image_h, class_num, keypoint_num, conf_thresh, nms_thresh
        );
    );
    
    checkCudaRuntime(cudaMemcpy(num_dets, num_detections, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaMemcpy(det_classes, detection_classes, sizeof(int) * keepTopK, cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaMemcpy(det_scores, detection_scores, sizeof(float) * keepTopK, cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaMemcpy(det_boxes, detection_boxes, sizeof(float) * keepTopK * 4, cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaMemcpy(det_keypoints, detection_keypoints, sizeof(float) * keepTopK * 3 * keypoint_num, cudaMemcpyDeviceToHost));

    checkCudaRuntime(cudaFree(detection_keypoints));
    checkCudaRuntime(cudaFree(detection_boxes));
    checkCudaRuntime(cudaFree(detection_scores));
    checkCudaRuntime(cudaFree(detection_classes));
    checkCudaRuntime(cudaFree(num_detections));

    checkCudaRuntime(cudaFree(d_keep));
    checkCudaRuntime(cudaFree(d_objectCount));
    checkCudaRuntime(cudaFree(d_output_objects));
    checkCudaRuntime(cudaFree(d_pBlob_offset));
    checkCudaRuntime(cudaFree(d_output_data));
}