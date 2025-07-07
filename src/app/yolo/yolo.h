/**
 * *****************************************************************************
 * File name:   yolo.h
 * 
 * @brief  YOLO's Model Inference
 * 支持yolox, yolov8
 * 
 * 
 * Created by Aegon on 2025-07-07
 * Copyright © 2025 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACINFER_ULTRA_YOLO_H
#define ACINFER_ULTRA_YOLO_H

#include <memory>
#include <opencv2/opencv.hpp>

#include <types/error.h>
#include "yolo_type.h"

namespace YOLO {

    enum class Type : int{
        X  = 0,
        V8 = 1
    };

    enum class DecodeMethod : int{
        CPU = 0,
        GPU = 1 
    };

    enum class Task : int {
        YOLO_DET   = 0,
        YOLO_SEG   = 1,
        YOLO_POSE  = 2
    };

    class Model {
    public:
        virtual error_e Run(const cv::Mat& frame, YoloResults& resluts) = 0;
    };

    std::shared_ptr<Model> CreateInferModel(
        const std::string& model_path,
        const Type yolo_type,
        const Task task_type = Task::YOLO_DET,
        const DecodeMethod decode_type = DecodeMethod::CPU,
        bool use_plugin = false
    );
    
} // namespace YOLO


#endif // ACINFER_ULTRA_YOLO_H