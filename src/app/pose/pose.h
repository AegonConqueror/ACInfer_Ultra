#ifndef ACINFER_ULTRA_POSE_H
#define ACINFER_ULTRA_POSE_H

#include <memory>
#include <types/error.h>

#include "yolo/yolov8_type.h"

typedef struct PosePoint {
    float x, y;
    float score;
} PosePoint;

namespace Pose {
    class Model {
    public:
        virtual error_e Run(const cv::Mat& frame, std::vector<yolov8_result>& det_results) = 0;
    };

    std::shared_ptr<Model> CreateInferModel(
        const std::string& pose_model_path,
        bool use_plugin = false
    );
} // namespace Pose


#endif // ACINFER_ULTRA_POSE_H