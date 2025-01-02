
#ifndef ACINFER_ULTRA_YOLOV8_POSTPROCESS_H
#define ACINFER_ULTRA_YOLOV8_POSTPROCESS_H

#include <stdint.h>
#include <vector>

#include "yolo/yolov8_type.h"

namespace yolov8 {

    void PostprocessSplit_DET(
        float **preds, std::vector<float> &det_rects,
        int input_w, int input_h, int class_num,
        float conf_thres = 0.45, float nms_thres = 0.45
    );

    void PostprocessSplit_POSE(
        float **preds, std::vector<float> &pose_rects,
        std::vector<std::map<int, KeyPoint>> &key_points,
        int input_w, int input_h, int class_num,
        float conf_thres = 0.45, float nms_thres = 0.45
    );

    void PostprocessSplit_SEG(
        float **preds, std::vector<float> &seg_rects,
        std::vector<cv::Mat> &seg_masks,
        int input_w, int input_h, int class_num,
        float conf_thres = 0.45, float nms_thres = 0.45
    );

} // namespace yolov8

#endif // ACINFER_ULTRA_YOLOV8_POSTPROCESS_H