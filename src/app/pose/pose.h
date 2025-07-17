#ifndef ACINFER_ULTRA_POSE_H
#define ACINFER_ULTRA_POSE_H

#include <memory>
#include <opencv2/opencv.hpp>

#include <types/error.h>

#include "yolo/yolo_type.h"

namespace Pose {
    class Model {
    public:
        virtual error_e Run(const cv::Mat& frame, std::vector<yolo_result>& det_results) = 0;
    };

    std::shared_ptr<Model> CreateInferModel(
        const std::string& pose_model_path,
        bool use_plugin = false
    );
} // namespace Pose


#endif // ACINFER_ULTRA_POSE_H