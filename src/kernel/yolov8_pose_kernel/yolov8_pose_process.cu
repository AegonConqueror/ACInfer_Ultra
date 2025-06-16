#include <cuda_runtime.h>

#include "trt/trt_cuda.h"

#include "yolo/yolov8_type.h"

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
void yolov8_pose_decode_kernel(
    float* pBlob, int* pBlob_offset, int* outputRects, float* outputSocres, int* obj_count, int* keep,
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
    
    float* reg = pBlob + pBlob_offset[head_idx * 2 + 0] + imageIdx * total_size;
    float* cls = pBlob + pBlob_offset[head_idx * 2 + 1] + imageIdx * total_size;
    float* ps  = pBlob + pBlob_offset[head_idx + headNum * 2] + imageIdx * total_size;

    // if (tid == 1000) {
    //     printf("reg %d, %f -- %f\n", total_anchors, pBlob[1000], pBlob[1000 + imageIdx * total_size]);
    // }
    
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

    int batch_offset = imageIdx * total_anchors;
    int id = atomicAdd(&obj_count[imageIdx], 1);

    int index_i = id + batch_offset;

    keep[index_i] = 1;

    int* index_i_rect = outputRects + index_i * (4 + 1);
    float* index_i_score = outputSocres + index_i;
    index_i_rect[0] = int(xmin * image_w / input_w + 0.5);
    index_i_rect[1] = int(ymin * image_h / input_h + 0.5);
    index_i_rect[2] = int(xmax * image_w / input_w + 0.5);
    index_i_rect[3] = int(ymax * image_h / input_h + 0.5);
    index_i_rect[4] = cls_index;
    index_i_score[0] = cls_max;

    __syncthreads();

    for (int j = 0; j < obj_count[imageIdx]; j++) {
        int index_j = j + batch_offset;

        int* index_j_rect = outputRects + index_j * (4 + 1);
        float* index_j_score = outputSocres + index_j;

        if (index_i == index_j || keep[index_j] == 0) continue;
        if (
            index_i_score[0] < index_j_score[0] && 
            index_i_rect[4] == index_j_rect[4]
        ) {
            float iou = iou_gpu(
                index_i_rect[0], index_i_rect[1], index_i_rect[2], index_i_rect[3], 
                index_j_rect[0], index_j_rect[1], index_j_rect[2], index_j_rect[3]
            );
            if (iou > nms_thresh) {
                keep[index_i] = 0;
            }
        }
    }

    __syncthreads();

    if (keep[index_i] == 1) {
        // NumDetections
        int kid = atomicAdd(&objectNum[imageIdx], 1);

        // DetectionClasses
        det_classes[kid + imageIdx * 20] = index_i_rect[4];

        // DetectionScores
        det_scores[kid + imageIdx * 20] = index_i_score[0];

        // DetectionBoxes
        det_boxes[(kid + imageIdx * 20) * 4 + 0] = index_i_rect[0];
        det_boxes[(kid + imageIdx * 20) * 4 + 1] = index_i_rect[1];
        det_boxes[(kid + imageIdx * 20) * 4 + 2] = index_i_rect[2];
        det_boxes[(kid + imageIdx * 20) * 4 + 3] = index_i_rect[3];

        // DetectionKeyPoints
        // float* det_keypoints_batch = det_keypoints + imageIdx * 20 * 3 * num_keypoints;

        for (int kc = 0; kc < num_keypoints; kc++) {
            det_keypoints[(kc + imageIdx * 20) * num_keypoints * 3 + kid * 3 + 0] = (ps[(kc * 3 + 0) * h * w + row * w + col] * 2 + (cx - 0.5f)) * stride * image_w / input_w;
            det_keypoints[(kc + imageIdx * 20) * num_keypoints * 3 + kid * 3 + 1] = (ps[(kc * 3 + 1) * h * w + row * w + col] * 2 + (cy - 0.5f)) * stride * image_h / input_h;
            det_keypoints[(kc + imageIdx * 20) * num_keypoints * 3 + kid * 3 + 2] = sigmoid_gpu(ps[(kc * 3 + 2) * h * w + row * w + col]);
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

    int batchSize = 2;

    float* d_output_data = nullptr;
    checkCudaRuntime(cudaMalloc(&d_output_data, total_size * sizeof(float) * batchSize));
    checkCudaRuntime(cudaMemcpy(d_output_data, output_data, total_size * sizeof(float) * batchSize, cudaMemcpyHostToDevice));
    
    int* d_pBlob_offset = nullptr;
    checkCudaRuntime(cudaMalloc(&d_pBlob_offset, output_num * sizeof(int)));
    checkCudaRuntime(cudaMemcpy(d_pBlob_offset, pBlob_offset, output_num * sizeof(int), cudaMemcpyHostToDevice));
    
    int* d_output_rects;
    float* d_output_scores;
    int* d_objectCount;
    int* d_keep;
    checkCudaRuntime(cudaMalloc(&d_output_rects, (4 + 1) * sizeof(int) * total_anchors * batchSize));
    checkCudaRuntime(cudaMemset(d_output_rects, 0, (4 + 1) * sizeof(int) * total_anchors * batchSize));
    checkCudaRuntime(cudaMalloc(&d_output_scores, sizeof(float) * total_anchors * batchSize));
    checkCudaRuntime(cudaMemset(d_output_scores, 0, sizeof(float) * total_anchors * batchSize));
    checkCudaRuntime(cudaMalloc(&d_objectCount, sizeof(int) * batchSize));
    checkCudaRuntime(cudaMemset(d_objectCount, 0, sizeof(int) * batchSize));
    checkCudaRuntime(cudaMalloc(&d_keep, sizeof(int) * total_anchors * batchSize));
    checkCudaRuntime(cudaMemset(d_keep, -1, sizeof(int) * total_anchors * batchSize));

    int keepTopK = 20;

    int* num_detections;
    int* detection_classes;
    float* detection_scores;
    float* detection_boxes;
    float* detection_keypoints;

    checkCudaRuntime(cudaMalloc(&num_detections, sizeof(int) * batchSize));
    checkCudaRuntime(cudaMalloc(&detection_classes, sizeof(int) * keepTopK * batchSize));
    checkCudaRuntime(cudaMalloc(&detection_scores, sizeof(float) * keepTopK * batchSize));
    checkCudaRuntime(cudaMalloc(&detection_boxes, sizeof(float) * keepTopK * 4 * batchSize));
    checkCudaRuntime(cudaMalloc(&detection_keypoints, sizeof(float) * keepTopK * 3 * keypoint_num * batchSize));

    checkCudaRuntime(cudaMemset(num_detections, 0, sizeof(int) * batchSize));
    checkCudaRuntime(cudaMemset(detection_classes, 0, sizeof(int) * keepTopK * batchSize));
    checkCudaRuntime(cudaMemset(detection_scores, 0, sizeof(float) * keepTopK * batchSize));
    checkCudaRuntime(cudaMemset(detection_boxes, 0, sizeof(float) * keepTopK * 4 * batchSize));
    checkCudaRuntime(cudaMemset(detection_keypoints, 0, sizeof(float) * keepTopK * 3 * keypoint_num * batchSize));

    int threadSize = 256;

    dim3 block(threadSize, 1);
    dim3 grid(batchSize, (total_anchors + threadSize - 1) / threadSize);

    checkCudaKernel(
        yolov8_pose_decode_kernel<<<grid, block>>>(
            d_output_data, d_pBlob_offset, d_output_rects, d_output_scores, d_objectCount, d_keep, 
            num_detections, detection_classes, detection_scores, detection_boxes, detection_keypoints,
            total_size, headNum, total_anchors, head_start[1], head_start[2], min_stride, 
            input_w, input_h, image_w, image_h, class_num, keypoint_num, conf_thresh, nms_thresh
        );
    );
    
    checkCudaRuntime(cudaMemcpy(num_dets, num_detections, sizeof(int) * batchSize, cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaMemcpy(det_classes, detection_classes, sizeof(int) * keepTopK * batchSize, cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaMemcpy(det_scores, detection_scores, sizeof(float) * keepTopK * batchSize, cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaMemcpy(det_boxes, detection_boxes, sizeof(float) * keepTopK * 4 * batchSize, cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaMemcpy(det_keypoints, detection_keypoints, sizeof(float) * keepTopK * 3 * keypoint_num * batchSize, cudaMemcpyDeviceToHost));

    checkCudaRuntime(cudaFree(detection_keypoints));
    checkCudaRuntime(cudaFree(detection_boxes));
    checkCudaRuntime(cudaFree(detection_scores));
    checkCudaRuntime(cudaFree(detection_classes));
    checkCudaRuntime(cudaFree(num_detections));

    checkCudaRuntime(cudaFree(d_keep));
    checkCudaRuntime(cudaFree(d_objectCount));
    checkCudaRuntime(cudaFree(d_output_scores));
    checkCudaRuntime(cudaFree(d_output_rects));
    checkCudaRuntime(cudaFree(d_pBlob_offset));
    checkCudaRuntime(cudaFree(d_output_data));
}