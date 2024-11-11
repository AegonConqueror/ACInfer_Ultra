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
#include "engine/engine_api.h"

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
        Detector(
            const std::string &model_path, 
            int platform, 
            Type yolo_type=Type::V8,
            bool use_plugins=false
        );
        ~Detector();
        error_e Run(const cv::Mat &frame, std::vector<detect_result> &objects);

    private:
        error_e Preprocess(const cv::Mat &frame, cv::Mat &timg);
        error_e Postprocess(const cv::Mat &frame, std::vector<detect_result> &objects);

    private:
        bool    use_plugins_;
        void*   engine_;
        Type    yolo_type_;

        std::vector<int> input_shape_;
    };
    
} // namespace YOLO

#endif // ACINFER_ULTRA_YOLO_H
