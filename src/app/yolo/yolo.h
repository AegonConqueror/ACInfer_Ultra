/**
 * *****************************************************************************
 * File name:   yolo.h
 * 
 * @brief  YOLO 
 * 
 * 
 * Created by Aegon on 2025-09-30
 * Copyright © 2025 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACINFER_ULTRA_YOLO_H
#define ACINFER_ULTRA_YOLO_H

#include <types/error.h>
#include "object_detector.h"

namespace YOLO {

    using namespace ObjectDetector;

    class Model {
    public:
        virtual ac_error_e Run(const cv::Mat& frame, DETResults& results)  = 0;
    };

    std::shared_ptr<Model> CreateInference(
        const std::string& model_path
    );
    
} // namespace YOLO

#endif // ACINFER_ULTRA_YOLO_H