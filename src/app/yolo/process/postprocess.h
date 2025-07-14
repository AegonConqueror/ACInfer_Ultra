#ifndef ACINFER_ULTRA_POSTPROCESS_H
#define ACINFER_ULTRA_POSTPROCESS_H

#include <stdint.h>
#include <vector>

#include "yolo/yolo_type.h"

namespace yolov8 {
    void Postprocess_POSE_float(
        float* reg, float* cls, float *ps, std::vector<float>& pose_rects, KeyPointsArray& key_points,
        int anchors, int input_w, int input_h, int class_num, int keypoint_num,
        float conf_thres = 0.45, float nms_thres = 0.45
    );
} // namespace yolov8


#endif // ACINFER_ULTRA_POSTPROCESS_H