#ifndef ACINFER_ULTRA_DWPOSE_H
#define ACINFER_ULTRA_DWPOSE_H

#include <memory>
#include <types/error.h>

#include <opencv2/opencv.hpp>

typedef struct KeyPoint {
    float x, y;
    float score;
} KeyPoint;

namespace DWPose {
    class Model {
    public:
        virtual error_e Run(const cv::Mat &frame, std::vector<KeyPoint>& results) = 0;
    };

    std::shared_ptr<Model> CreateInferModel(
        const std::string &model_path,
        bool use_plugin = false
    );
} // namespace DWPose


#endif // ACINFER_ULTRA_DWPOSE_H