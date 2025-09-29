/**
 * *****************************************************************************
 * File name:   pose_detector.h
 * 
 * @brief  人型关键点检测类算法相关结构体
 * 
 * 
 * Created by Aegon on 2025-08-06
 * Copyright © 2025 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACINFER_ULTRA_POSE_DETECTOR_H
#define ACINFER_ULTRA_POSE_DETECTOR_H

#include <vector>
#include <map>

#include "object_detector.h"

namespace PoseDetector {

    using namespace ObjectDetector;

    struct KeyPoint {
        int id;
        float x, y, score;

        KeyPoint() = default;
        KeyPoint(int id, float x, float y, float score): id(id), x(x), y(y), score(score) {}
    };
    typedef std::map<int, KeyPoint> KeyPoints;
    typedef std::vector<KeyPoints> KeyPointsArray;

    struct POSEResult : public DETResult {
        KeyPoints keypoints;

        POSEResult() = default;
        POSEResult(int classId, float score, const Box& box, const KeyPoints& keypoints) 
            :DETResult(classId, score, box), keypoints(keypoints){}
        POSEResult(const DETResult& object, const KeyPoints& keypoints) 
            :DETResult(object.classId, object.score, object.box), keypoints(keypoints){}
    };
    typedef std::vector<POSEResult> POSEResults;
    
} // namespace PoseDetector


#endif // ACINFER_ULTRA_POSE_DETECTOR_H