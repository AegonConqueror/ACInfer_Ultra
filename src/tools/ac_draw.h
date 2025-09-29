/**
 * *****************************************************************************
 * File name:   ac_draw.h
 * 
 * @brief  
 * 
 * 
 * Created by Aegon on 2023-04-18
 * Copyright © 2023 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACINFER_ULTRA_AC_DRAW_H
#define ACINFER_ULTRA_AC_DRAW_H

#include <opencv2/opencv.hpp>

namespace iDraw {

    void draw_box(cv::Mat &img, const cv::Rect_<float> &box, const int class_id);

    void draw_box_label(
        cv::Mat& img, const cv::Rect_<float> box, 
        const int class_id, const float conf,
        const std::vector<std::string>& class_names={}
    );

    void draw_mask(cv::Mat& img, const cv::Rect_<float>& box, cv::Mat& roi);
} // namespace iDra

#endif // ACINFER_ULTRA_AC_DRAW_H