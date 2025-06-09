#include <cuda_runtime.h>

#include "trt/trt_cuda.h"

#include "yolo/yolov8_type.h"

struct DetectRectGPU {
    float xmin, ymin, xmax, ymax;
    int classId;
    float score;
};

__device__ 
float sigmoid_gpu(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__
float iou_gpu(const DetectRectGPU &a, const DetectRectGPU &b) {
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
void nms_kernel(const DetectRectGPU* dets, int* keep, int num, float iou_thresh) {
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
void generate_meshgrid_kernel(float* meshgrid, const int* mapSize, int headNum){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_idx = 0;
    for (int head = 0; head < headNum; head++) {
        int h = mapSize[head * 2];
        int w = mapSize[head * 2 + 1];
        int num = h * w;
        if (tid < total_idx + num) {
            int idx = tid - total_idx;
            int row = idx / w;
            int col = idx % w;
            meshgrid[tid * 2 + 0] = col + 0.5f;
            meshgrid[tid * 2 + 1] = row + 0.5f;
            return;
        }
        total_idx += num;
    }
}

__global__
void yolov8_det_decode_kernel(
    float** preds, float* meshgrid, DetectRectGPU* outputRects, int* obj_count, 
    int* map_size, int* strides, int* head_start, int total_anchors, int headNum, 
    int input_w, int input_h, int classNum, float confThresh
){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_anchors) return;

    int head_idx = -1;
    for (int i = 0; i < headNum; i++) {
        int start = head_start[i];
        int end = (i + 1 < headNum) ? head_start[i + 1] : total_anchors;
        if (tid >= start && tid < end) {
            head_idx = i;
            break;
        }
    }

    if (head_idx == -1) return;

    int h = map_size[head_idx * 2 + 0];
    int w = map_size[head_idx * 2 + 1];
    int local_idx = tid - head_start[head_idx];
    int row = local_idx / w;
    int col = local_idx % w;
    int stride = strides[head_idx];
    
    float* reg = preds[head_idx * 2 + 0];
    float* cls = preds[head_idx * 2 + 1];

    float cx = meshgrid[tid * 2];
    float cy = meshgrid[tid * 2 + 1];

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

    DetectRectGPU& temp = outputRects[id];
    temp.xmin = xmin / input_w;
    temp.ymin = ymin / input_h;
    temp.xmax = xmax / input_w;
    temp.ymax = ymax / input_h;
    temp.classId = cls_index;
    temp.score = cls_max;
}

void yolov8_det_decode_gpu(
    float** preds, int* d_size, std::vector<float> &pose_rects,
    int input_w, int input_h, int class_num,
    float conf_thres, float nms_thres
) {
    int headNum = 3;
    int mapSize[headNum * 2] = {80, 80, 40, 40, 20, 20};
    int strides[headNum] = {8, 16, 32};
    int head_start[headNum];

    int total_anchors = 0;
    for (int i = 0; i < headNum; ++i){
        int total = mapSize[i * 2] * mapSize[i * 2 + 1];
        head_start[i] = total_anchors;
        total_anchors += total;
    }

    float* d_meshgrid;
    checkCudaRuntime(cudaMalloc(&d_meshgrid, sizeof(float) * total_anchors * 2));

    int* d_mapSize;
    int* d_strides;
    int* d_head_start;
    checkCudaRuntime(cudaMalloc(&d_mapSize, sizeof(int) * headNum * 2));
    checkCudaRuntime(cudaMalloc(&d_strides, sizeof(int) * headNum));
    checkCudaRuntime(cudaMalloc(&d_head_start, sizeof(int) * headNum));
    checkCudaRuntime(cudaMemcpy(d_mapSize, mapSize, sizeof(int) * headNum * 2, cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaMemcpy(d_strides, strides, sizeof(int) * headNum, cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaMemcpy(d_head_start, head_start, sizeof(int) * headNum, cudaMemcpyHostToDevice));

    int threads = 256;
    int blocks = (total_anchors + threads - 1) / threads;
    checkCudaKernel(
        generate_meshgrid_kernel<<<blocks, threads>>>(d_meshgrid, d_mapSize, headNum);
    );

    std::vector<float> h_meshgrid(total_anchors * 2);
    checkCudaRuntime(cudaMemcpy(h_meshgrid.data(), d_meshgrid, sizeof(float) * total_anchors * 2, cudaMemcpyDeviceToHost));
    
    DetectRectGPU* d_output_objects;
    int* d_objectCount;
    checkCudaRuntime(cudaMalloc(&d_output_objects, sizeof(DetectRectGPU) * total_anchors));
    checkCudaRuntime(cudaMalloc(&d_objectCount, sizeof(int)));
    checkCudaRuntime(cudaMemset(d_objectCount, 0, sizeof(int)));

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

    checkCudaKernel(
        yolov8_det_decode_kernel<<<blocks, threads>>>(
            pBlob_d, d_meshgrid, d_output_objects, d_objectCount, d_mapSize, d_strides, 
            d_head_start, total_anchors, headNum, input_w, input_h, class_num, conf_thres
        );
    );

    int object_num;
    checkCudaRuntime(cudaMemcpy(&object_num, d_objectCount, sizeof(int), cudaMemcpyDeviceToHost));

    int* d_keep;
    checkCudaRuntime(cudaMalloc(&d_keep, sizeof(int) * object_num));
    checkCudaRuntime(cudaMemset(d_keep, 1, sizeof(int) * object_num));
    
    checkCudaKernel(
        nms_kernel<<<blocks, threads>>>(d_output_objects, d_keep, object_num, nms_thres);
    );

    std::vector<DetectRectGPU> h_objects(object_num);
    std::vector<int> h_keep(object_num);
    checkCudaRuntime(cudaMemcpy(h_objects.data(), d_output_objects, sizeof(DetectRectGPU) * object_num, cudaMemcpyDeviceToHost));
    checkCudaRuntime(cudaMemcpy(h_keep.data(), d_keep, sizeof(int) * object_num, cudaMemcpyDeviceToHost));

    for (int i = 0; i < object_num; i++) {
        if (h_keep[i] == 0) continue;
        auto& obj = h_objects[i];
        pose_rects.push_back((float)obj.classId);
        pose_rects.push_back((float)obj.score);
        pose_rects.push_back(obj.xmin);
        pose_rects.push_back(obj.ymin);
        pose_rects.push_back(obj.xmax);
        pose_rects.push_back(obj.ymax);
    }


    for (int i = 0; i < blob_num; ++i) {
        cudaFree(pBlob_d_host[i]);
    }
    delete[] pBlob_d_host;
    checkCudaRuntime(cudaFree(pBlob_d));
    checkCudaRuntime(cudaFree(d_keep));
    checkCudaRuntime(cudaFree(d_objectCount));
    checkCudaRuntime(cudaFree(d_output_objects));
    checkCudaRuntime(cudaFree(d_strides));
    checkCudaRuntime(cudaFree(d_mapSize));
    checkCudaRuntime(cudaFree(d_meshgrid)); 
}