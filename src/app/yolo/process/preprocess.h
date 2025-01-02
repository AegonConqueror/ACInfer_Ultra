
#ifndef ACINFER_ULTRA_PREPROCESS_H
#define ACINFER_ULTRA_PREPROCESS_H

#include <opencv2/opencv.hpp>

struct LetterBoxInfo {
    bool hor;
    int pad;
};

LetterBoxInfo letterbox(const cv::Mat &img, cv::Mat &img_letterbox, float wh_ratio);

#endif // ACINFER_ULTRA_PREPROCESS_H