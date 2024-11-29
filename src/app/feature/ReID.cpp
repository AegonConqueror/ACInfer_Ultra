
#include "ReID.h"

class ReIDModelImpl : public ReIDModel {
public:
    virtual Eigen::Matrix<float, 1, 512> get_features(cv::Mat &image) override;
    error_e         load(const std::string &onnx_model_path);
    void            preprocess(cv::Mat &image);

private:
    std::vector<int>            input_shape_;
    std::shared_ptr<ACEngine>   engine_;
};

error_e ReIDModelImpl::load(const std::string &onnx_model_path) {
    engine_ = create_engine(onnx_model_path, true);
    if (!engine_) {
        LOG_ERROR("Deserialize onnx failed.");
        return MODEL_NOT_LOAD;
    }
    engine_->Print();
    input_shape_ = engine_->GetInputShape();
    return SUCCESS;
}

Eigen::Matrix<float, 1, 512> ReIDModelImpl::get_features(cv::Mat &image_patch){
    preprocess(image_patch);

    InferenceDataType infer_result_data;
    engine_->GetInferOutput(infer_result_data);

    auto output_shape = engine_->GetOutputShapes()[0];
    if (output_shape[0] == -1) {
        output_shape[0] = 1;
    }
    
    int output_size = output_shape[1];
    float* pred = nullptr;
    bool is_float16_allocated = false;

    if (engine_->GetInputType() == "Float16"){
        uint16_t *item = (uint16_t *)infer_result_data[0].first;
        pred = iTools::halfToFloat((void *)item, output_shape);
        is_float16_allocated = true;
    } else {
        pred = (float *)infer_result_data[0].first;
    }

    Eigen::Matrix<float, 1, 512>  feature_vector = Eigen::Matrix<float, 1, 512>::Zero(1, output_size);
    for (int i = 0; i < output_size; i++) {
        feature_vector(0, i) = pred[i];
    }

    // 释放内存
    if (is_float16_allocated) {
        delete[] pred;  // 释放通过 halfToFloat 分配的内存
    }
    return feature_vector;
}


void ReIDModelImpl::preprocess(cv::Mat &image) {
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    cv::resize(
        image, resizedImageBGR, 
        cv::Size(input_shape_[3], input_shape_[2]), 
        cv::InterpolationFlags::INTER_CUBIC
    );
    cv::cvtColor(resizedImageBGR, resizedImageRGB, cv::ColorConversionCodes::COLOR_BGR2RGB);

    resizedImage = resizedImageRGB / 255.0;
    cv::Mat channels[3];
    cv::split(resizedImage, channels);
    //  Normalization per channel
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    cv::merge(channels, 3, resizedImage);
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);
    int targetType = (engine_->GetInputType() == "Float") ? CV_32F : CV_16F;
    preprocessedImage.convertTo(preprocessedImage, targetType);

    uint32_t fileSize = preprocessedImage.total() * preprocessedImage.elemSize();
    InferenceDataType infer_input_data;

    if (engine_->GetInputType() == "Float16"){
        infer_input_data.push_back(
             std::make_pair((void*)preprocessedImage.ptr<uint16_t>(), fileSize)
        );
    } else {
        infer_input_data.push_back(
             std::make_pair((void*)preprocessedImage.ptr<float>(), fileSize)
        );
    }
    engine_->BindingInput(infer_input_data);

}

std::shared_ptr<ReIDModel> CreateReID(
    const std::string &model_path
) {
    std::shared_ptr<ReIDModelImpl> Instance(new ReIDModelImpl());
    if (Instance->load(model_path) != SUCCESS) {
        Instance.reset();
    }
    return Instance;
}