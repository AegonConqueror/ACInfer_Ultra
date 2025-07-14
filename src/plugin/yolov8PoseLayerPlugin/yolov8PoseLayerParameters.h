#ifndef ACINFER_ULTRA_YOLOV8POSELAYERPARAMETERS_H
#define ACINFER_ULTRA_YOLOV8POSELAYERPARAMETERS_H

#include "common/plugin.h"

struct YOLOv8PoseLayerParameters {

    int batchSize = 1;

    int inputWidth = 640;
    int inputHeight = 640;

    int numOutputBoxes = 20;

    int numClasses = 1;
    int numKeypoints = 17;
    int numAnchors = 8400;

    int minStride = 8;

    float iouThreshold = 0.45f;
    float scoreThreshold = 0.25f;
};

#endif // ACINFER_ULTRA_YOLOV8POSELAYERPARAMETERS_H