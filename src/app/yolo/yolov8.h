
#ifndef ACINFER_ULTRA_YOLOV8_H
#define ACINFER_ULTRA_YOLOV8_H

#include <memory>
#include <types/error.h>

#include "yolov8_type.h"

namespace YOLOv8 {

    class Model {
    public:
        virtual error_e Run(const cv::Mat &frame, std::vector<yolov8_result> &objects) = 0;
    };

    std::shared_ptr<Model> CreateInferModel(
        const std::string &model_path, 
        const TaskType task_type = TaskType::YOLOv8_DET,
        bool use_plugin = false
    );
    
} // namespace YOLOv8


#endif // ACINFER_ULTRA_YOLOV8_H