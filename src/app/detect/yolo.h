/**
 * *****************************************************************************
 * File name:   yolo.h
 * 
 * @brief  YOLO 目标检测
 * 
 * 
 * Created by Aegon on 2023-06-30
 * Copyright © 2023 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACINFER_ULTRA_YOLO_H
#define ACINFER_ULTRA_YOLO_H

#include "types/ACType.h"
#include "engine/engine.h"

namespace YOLO {

    typedef struct detect_result {
        cv::Rect_<float>    box;
        int                 class_id;
        float               confidence;
    } detect_result;

    enum class Type : int{
        V8 = 0
    };

    class Detector {
    public:
        virtual error_e Run(const cv::Mat &frame, std::vector<detect_result> &objects) = 0;
    };

    std::shared_ptr<Detector> CreateDetector(
        const std::string &model_path, 
        Type yolo_type=Type::V8,
        bool use_plugins=false
    );
    
} // namespace YOLO

#endif // ACINFER_ULTRA_YOLO_H
