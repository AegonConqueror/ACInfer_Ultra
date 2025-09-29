/**
 * *****************************************************************************
 * File name:   object_detector.h
 * 
 * @brief  目标检测类算法相关结构体
 * 
 * 
 * Created by Aegon on 2025-08-06
 * Copyright © 2025 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACINFER_ULTRA_OBJECT_DETECTOR_H
#define ACINFER_ULTRA_OBJECT_DETECTOR_H

#include <opencv2/opencv.hpp>

namespace ObjectDetector {
    struct Box{
        int x, y, width, height;

        Box() = default;
        Box(int x, int y, int w, int h) :x(x), y(y), width(w), height(h){}

        operator cv::Rect() const {
            return cv::Rect(x, y, width, height);
        }
    };

    struct DETResult {
        int     classId;
        float   score;
        Box     box;
        
        DETResult() = default;
        DETResult(int classId, float score, const Box& box) :classId(classId), score(score), box(box){}
    };
    typedef std::vector<DETResult> DETResults;

} // namespace ObjectDetector

#endif // ACINFER_ULTRA_OBJECT_DETECTOR_H