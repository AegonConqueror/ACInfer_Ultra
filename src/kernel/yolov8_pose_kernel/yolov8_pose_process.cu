#include <cuda_runtime.h>

#include "trt/trt_cuda.h"

#include "yolo/yolov8_type.h"

struct KeyPointGPU {
    float x, y;
    float score;
};

struct PoseRectGPU {
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
void nms_kernel(
    const PoseRectGPU* dets, int* keep, int* objectCount, 
    int* objectNum, int* det_classes, float* det_scores, float* det_boxes, float* det_keypoints,
    int keypoint_num, float iou_thresh
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= *objectCount || keep[idx] == 0) return;

    for (int j = 0; j < *objectCount; j++) {
        if (idx == j || keep[j] == 0) continue;
        if (dets[idx].score < dets[j].score && dets[idx].classId == dets[j].classId) {
            float iou = iou_gpu(dets[idx], dets[j]);
            if (iou > iou_thresh) {
                keep[idx] = 0;
            }
        }
    }

    __syncthreads();

    if (keep[idx] == 1) {
        // NumDetections
        int id = atomicAdd(objectNum, 1);

        // DetectionClasses
        det_classes[id] = dets[idx].classId;

        // DetectionScores
        det_scores[id] = dets[idx].score;

        // DetectionBoxes
        det_boxes[id * 4 + 0] = dets[idx].xmin;
        det_boxes[id * 4 + 1] = dets[idx].ymin;
        det_boxes[id * 4 + 2] = dets[idx].xmax;
        det_boxes[id * 4 + 3] = dets[idx].ymax;

        // DetectionKeyPoints
        for (int k = 0; k < keypoint_num; k++) {
            det_keypoints[k * keypoint_num * 3 + id * 3 + 0] = dets[idx].keypoints[k].x;
            det_keypoints[k * keypoint_num * 3 + id * 3 + 1] = dets[idx].keypoints[k].y;
            det_keypoints[k * keypoint_num * 3 + id * 3 + 2] = dets[idx].keypoints[k].score;
        }
    }
}

__global__
void yolov8_pose_decode_kernel(
    float** preds, PoseRectGPU* outputRects, int* obj_count, int* keep, int headNum, 
    int total_anchors, int head_start, int head_end, int min_stride, int input_w, 
    int input_h, int image_w, int image_h, int classNum, int kepPointNum, float confThresh
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_anchors) return;

    int head_idx = (tid < head_start) ? 0 : ((tid < head_end) ? 1 : 2);
    int local_start = (head_idx == 1) * head_start + (head_idx == 2) * head_end;

    int stride = min_stride << head_idx;
    
    int h = input_h / stride;
    int w = input_w / stride;
    int local_idx = tid - local_start;
    int row = local_idx / w;
    int col = local_idx % w;
    
    float* reg = preds[head_idx * 2 + 0];
    float* cls = preds[head_idx * 2 + 1];
    float* ps  = preds[head_idx + headNum * 2];

    float cx = float(col + 0.5);
    float cy = float(row + 0.5);

    float cls_max = -1;
    int cls_index = -1;
    for (int cl = 0; cl < classNum; cl++) {
        float cls_val = sigmoid_gpu(cls[cl * h * w + row * w + col]);
        if (cls_val > cls_max) {
            cls_max = cls_val;
            cls_index = cl;
        }
    }

    if (cls_max < confThresh) return;

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
    keep[id] = 1;

    PoseRectGPU& temp = outputRects[id];
    temp.xmin = int(xmin * image_w / input_w + 0.5);
    temp.ymin = int(ymin * image_h / input_h + 0.5);
    temp.xmax = int(xmax * image_w / input_w + 0.5);
    temp.ymax = int(ymax * image_h / input_h + 0.5);
    temp.classId = cls_index;
    temp.score = cls_max;

    for (int kc = 0; kc < kepPointNum; kc++) {
        temp.keypoints[kc].x = (ps[(kc * 3 + 0) * h * w + row * w + col] * 2 + (cx - 0.5f)) * stride * image_w / input_w;
        temp.keypoints[kc].y = (ps[(kc * 3 + 1) * h * w + row * w + col] * 2 + (cy - 0.5f)) * stride * image_h / input_h;
        temp.keypoints[kc].score = sigmoid_gpu(ps[(kc * 3 + 2) * h * w + row * w + col]);
    }
}

void yolov8_pose_decode_gpu(
    float** preds, int* d_size, 
    int* num_dets, int* det_classes, float* det_scores, float* det_boxes, float* det_keypoints,
    int input_w, int input_h, int image_w, int image_h, int class_num,
    float conf_thres, float nms_thres, int keypoint_num
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

    int blob_num = headNum * 2 + headNum;
    float** h_pBlob = new float*[blob_num];
    float** pBlob_d_host = new float*[blob_num];

    for (int i = 0; i < blob_num; ++i) {
        size_t tensor_size = d_size[i];
        float* device_data = nullptr;
        checkCudaRuntime(cudaMalloc(&device_data, tensor_size * sizeof(float)));
        checkCudaRuntime(cudaMemcpy(device_data, preds[i], tensor_size * sizeof(float), cudaMemcpyHostToDevice));
        pBlob_d_host[i] = device_data;
    }

    float** pBlob_d = nullptr;
    checkCudaRuntime(cudaMalloc(&pBlob_d, blob_num * sizeof(float*)));
    checkCudaRuntime(cudaMemcpy(pBlob_d, pBlob_d_host, blob_num * sizeof(float*), cudaMemcpyHostToDevice));


    int threads = 256;
    int blocks = (total_anchors + threads - 1) / threads;
    
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

    auto time_start = iTime::timestamp_now_float();

    checkCudaKernel(
        yolov8_pose_decode_kernel<<<blocks, threads>>>(
            pBlob_d, d_output_objects, d_objectCount, d_keep, headNum, total_anchors, head_start[1], head_start[2], 
            min_stride, input_w, input_h, image_w, image_h, class_num, keypoint_num, conf_thres
        );
    );

    checkCudaKernel(
        nms_kernel<<<blocks, threads>>>(
            d_output_objects, d_keep, d_objectCount, 
            num_detections, detection_classes, detection_scores, detection_boxes, detection_keypoints,
            keypoint_num, nms_thres
        );
    );

    auto time_end = iTime::timestamp_now_float();
    LOG_INFO("use %f ms !", time_end - time_start);

    int h_num_dets[1];
    int h_det_classes[1 * keepTopK]; 
    float h_det_scores[1 * keepTopK]; 
    float h_det_boxes[1 * keepTopK * 4];
    float h_det_keypoints[1 * keepTopK * 3 * keypoint_num];
    
    checkCudaRuntime(cudaMemcpy(num_dets, num_detections, sizeof(int), cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaMemcpy(det_classes, detection_classes, sizeof(int) * keepTopK, cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaMemcpy(det_scores, detection_scores, sizeof(float) * keepTopK, cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaMemcpy(det_boxes, detection_boxes, sizeof(float) * keepTopK * 4, cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaMemcpy(det_keypoints, detection_keypoints, sizeof(float) * keepTopK * 3 * keypoint_num, cudaMemcpyDeviceToHost));

    checkCudaRuntime(cudaFree(num_detections));
    checkCudaRuntime(cudaFree(detection_classes));
    checkCudaRuntime(cudaFree(detection_scores));
    checkCudaRuntime(cudaFree(detection_boxes));
    checkCudaRuntime(cudaFree(detection_keypoints));
    
    for (int i = 0; i < blob_num; ++i) {
        cudaFree(pBlob_d_host[i]);
    }
    delete[] pBlob_d_host;
    checkCudaRuntime(cudaFree(pBlob_d));
    checkCudaRuntime(cudaFree(d_keep));
    checkCudaRuntime(cudaFree(d_objectCount));
    checkCudaRuntime(cudaFree(d_output_objects));
}