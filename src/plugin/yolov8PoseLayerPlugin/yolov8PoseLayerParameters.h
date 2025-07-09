#ifndef ACINFER_ULTRA_YOLOV8POSELAYERPARAMETERS_H
#define ACINFER_ULTRA_YOLOV8POSELAYERPARAMETERS_H

#include "common/plugin.h"

struct YOLOv8PoseLayerParameters {

    int inputWidth = 640;
    int inputHeight = 640;
    int numOutputBoxes = 20; ////

    int batchSize = 1;
    int numClasses = 1; //
    int numKeypoints = 17; //
    int numAnchors = 8400; // 
    int minStride = 8; ////

    float iouThreshold = 0.25f; ////
    float scoreThreshold = 0.45f; ////

    int headStart = 6400; //
    int headEnd = 8000;  //

    int reg1Size = 1; //
    int reg2Size = 1; //
    int reg3Size = 1; //

    int cls1Size = 1; //
    int cls2Size = 1; //
    int cls3Size = 1; //

    int ps1Size = 1; //
    int ps2Size = 1; //
    int ps3Size = 1; //
};

#endif // ACINFER_ULTRA_YOLOV8POSELAYERPARAMETERS_H