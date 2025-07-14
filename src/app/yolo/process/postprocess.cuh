#ifndef ACINFER_ULTRA_POSTPROCESS_CUH
#define ACINFER_ULTRA_POSTPROCESS_CUH

#include <stdint.h>
#include <vector>

#include "yolo/yolo_type.h"

#include "plugin/yolov8PoseLayerPlugin/yolov8PoseLayerParameters.h"

void YOLOv8PoseLayerInference(
    YOLOv8PoseLayerParameters param,
    float* regInput, float* clsInput, float* psInput,
    int regSize, int clsSize, int psSize,
    int* numDetectionsOutput, int* nmsClassesOutput, float* nmsScoresOutput, 
    float* nmsBoxesOutput, float* nmsKeyPointsOutput
);

#endif // ACINFER_ULTRA_POSTPROCESS_CUH