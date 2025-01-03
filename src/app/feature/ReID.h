
#ifndef ACINFER_ULTRA_REID_H
#define ACINFER_ULTRA_REID_H

#include "engine.h"
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include <opencv2/opencv.hpp>

class ReIDModel {
public:
    virtual Eigen::Matrix<float, 1, 512> get_features(cv::Mat &image) = 0;
};

std::shared_ptr<ReIDModel> CreateReID(
    const std::string &model_path
);

#endif // ACINFER_ULTRA_REID_H


