#ifndef ACINFER_ULTRA_YOLOV8DETLAYERPARAMETERS_H
#define ACINFER_ULTRA_YOLOV8DETLAYERPARAMETERS_H

#include "plugin.h"

struct YOLOv8DetLayerParameters {

    int batchSize = 1;

    int inputWidth = 640;
    int inputHeight = 640;

    int numOutputBoxes = 100;

    int numClasses = 1;
    int numAnchors = 8400;

    int minStride = 8;

    float iouThreshold = 0.45f;
    float scoreThreshold = 0.25f;
};

#endif // ACINFER_ULTRA_YOLOV8DETLAYERPARAMETERS_H