
#ifndef ACINFER_ULTRA_POSTPROCESS_H
#define ACINFER_ULTRA_POSTPROCESS_H

#include <vector>

typedef struct{
    float x;
    float y;
    float w;
    float h;
    float conf;
    int cls;
} Box;

namespace yolov8 {

    void PostprocessSpilt(
        float **pBlob, std::vector<float> &DetectiontRects,
        int input_width, int input_height, int class_num,
        float conf_thres = 0.5, float nms_thres = 0.45
    );

    void PostprocessNormal(
        std::vector<Box>& res, float *pred, int num_boxes, int num_class, 
        float conf_thres = 0.4, float nms_thres = 0.45
    );
} // namespace yolov8

#endif // ACINFER_ULTRA_POSTPROCESS_H