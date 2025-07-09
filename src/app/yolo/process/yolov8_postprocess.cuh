#ifndef ACINFER_ULTRA_YOLOV8_POSTPROCESS_CUH
#define ACINFER_ULTRA_YOLOV8_POSTPROCESS_CUH

#include <stdint.h>
#include <vector>

#include "yolo/yolo_type.h"

#include "plugin/yolov8PoseLayerPlugin/yolov8PoseLayerParameters.h"

void yolov8_pose_decode_gpu(
    float* output_data, int* output_size, int output_num,
    int* num_dets, int* det_classes, float* det_scores, float* det_boxes, float* det_keypoints,
    int input_w, int input_h, int image_w, int image_h, int class_num,
    float conf_thres, float nms_thres, int keypoint_num
);

void YOLOv8PoseLayerInference(
    YOLOv8PoseLayerParameters param,
    float* reg1Input, float* reg2Input, float* reg3Input,
    float* cls1Input, float* cls2Input, float* cls3Input,
    float* ps1Input, float* ps2Input, float* ps3Input,
    int* numDetectionsOutput, int* nmsClassesOutput, float* nmsScoresOutput, 
    float* nmsBoxesOutput, float* nmsKeyPointsOutput
);

#endif // ACINFER_ULTRA_YOLOV8_POSTPROCESS_CUH