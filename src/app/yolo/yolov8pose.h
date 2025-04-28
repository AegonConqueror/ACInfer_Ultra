
#ifndef ACINFER_ULTRA_YOLOV8POSE_H
#define ACINFER_ULTRA_YOLOV8POSE_H

#include <memory>
#include <types/error.h>

#include <opencv2/opencv.hpp>

typedef struct KeyPoint {
    float x, y;
    float score;
    int id;
} KeyPoint;

typedef struct posev8_result {
    cv::Rect                    box;
    int                         class_id;
    float                       confidence;
    std::map<int, KeyPoint>     keypoints;
} yolov8_result;


namespace POSEv8 {
    class Model {
    public:
        virtual error_e Run(const cv::Mat &frame, std::vector<yolov8_result> &objects) = 0;
    };

    std::shared_ptr<Model> CreateInferModel(
        const std::string &model_path,
        bool use_plugin = false
    );
} // namespace POSEv8


#endif // ACINFER_ULTRA_YOLOV8POSE_H