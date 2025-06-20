
#include "DWPose.h"

#include "engine.h"

#include "yolo/yolov8.h"
#include "pose.h"

cv::Mat resize_image(const cv::Mat& input_image, int target_resolution = 512, int dividable_by = 64) {
    int height = input_image.rows;
    int width = input_image.cols;

    float k = static_cast<float>(target_resolution) / std::min(height, width);

    int target_width = static_cast<int>(std::round((width * k) / dividable_by)) * dividable_by;
    int target_height = static_cast<int>(std::round((height * k) / dividable_by)) * dividable_by;

    int interpolation = (k > 1.0f) ? cv::INTER_LANCZOS4 : cv::INTER_AREA;

    cv::Mat resized_image;
    cv::resize(input_image, resized_image, cv::Size(target_width, target_height), 0, 0, interpolation);
    return resized_image;
}

namespace DWPose {

    class ModelImpl : public Model {
    public:
        error_e Load(const std::string& det_model_path, const std::string& pose_model_path);

        virtual error_e Run(const cv::Mat& frame, std::vector<yolov8_result>& results) override;
    private:

        error_e postprocess();

    private:
        std::shared_ptr<YOLOv8::Model> detector_;
        std::shared_ptr<Pose::Model> pose_;

    };

    error_e ModelImpl::Load(const std::string& det_model_path, const std::string& pose_model_path) {
        detector_ = YOLOv8::CreateInferModel(det_model_path, TaskType::YOLOv8_POSE);
        if (detector_ == nullptr) {
            LOG_ERROR("create detector failed.");
            return LOAD_MODEL_FAIL;
        }

        pose_ = Pose::CreateInferModel(pose_model_path);
        if (pose_ == nullptr) {
            LOG_ERROR("create pose failed.");
            return LOAD_MODEL_FAIL;
        }

        return SUCCESS;
    }

    error_e ModelImpl::postprocess() {
        return SUCCESS;
    }

    error_e ModelImpl::Run(const cv::Mat& frame, std::vector<yolov8_result>& results) {
        
        cv::Mat input_image = frame.clone();  // 复制原图
        int original_height = input_image.rows;
        int original_width = input_image.cols;

        input_image = resize_image(input_image, 640);
        int height = input_image.rows;
        int width = input_image.cols;


        detector_->Run(input_image, results);
        LOG_INFO("detector results %d", results.size());
        pose_->Run(input_image, results);

        return SUCCESS;
    }

    std::shared_ptr<Model> CreateInferModel(
        const std::string &det_model_path,
        const std::string &pose_model_path
    ) {
        std::shared_ptr<ModelImpl> Instance(new ModelImpl());
        if (Instance->Load(det_model_path, pose_model_path) != SUCCESS) {
            Instance.reset();
        }
        return Instance;
    }

} // namespace DWPose
