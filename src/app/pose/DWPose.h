#ifndef ACINFER_ULTRA_DWPOSE_H
#define ACINFER_ULTRA_DWPOSE_H

#include <memory>
#include <types/error.h>

#include "yolo/yolov8_type.h"

// typedef struct KeyPoint {
//     float x, y;
//     float score;
// } KeyPoint;

namespace DWPose {
    class Model {
    public:
        virtual error_e Run(const cv::Mat& frame, std::vector<yolov8_result>& results) = 0;
    };

    std::shared_ptr<Model> CreateInferModel(
        const std::string& det_model_path,
        const std::string& pose_model_path
    );
} // namespace DWPose


#endif // ACINFER_ULTRA_DWPOSE_H