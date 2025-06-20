#include "pose.h"

#include "engine.h"

typedef struct AffineImg {
    cv::Mat src_img;
    cv::Mat norm_img;
} AffineImg;

void pose_decode(
    const float* simcc_x,
    const float* simcc_y,
    std::map<int, KeyPoint>& pose_points,
    float simcc_split_ratio = 2.0f
) {
    int K = 133;
    int Wx = 576;
    int Wy = 786;

    for (int i = 0; i < K; ++i) {
        const float* px = simcc_x + i * Wx;
        const float* py = simcc_y + i * Wy;

        // 找最大值及索引
        int max_x_idx = 0;
        float max_x_val = px[0];
        for (int j = 1; j < Wx; ++j) {
            if (px[j] > max_x_val) {
                max_x_val = px[j];
                max_x_idx = j;
            }
        }

        int max_y_idx = 0;
        float max_y_val = py[0];
        for (int j = 1; j < Wy; ++j) {
            if (py[j] > max_y_val) {
                max_y_val = py[j];
                max_y_idx = j;
            }
        }

        float val = max_x_val > max_y_val ? max_y_val : max_x_val;

        KeyPoint kp;
        if (val > 0.0f) {
            kp.x = static_cast<float>(max_x_idx);
            kp.y = static_cast<float>(max_y_idx);
        } else {
            kp.x = -1.0f;
            kp.y = -1.0f;
        }

        kp.score = val;

        // 除以 split_ratio（若需要）
        kp.x /= simcc_split_ratio;
        kp.y /= simcc_split_ratio;
        kp.id = i;

        pose_points.insert({i, kp});
    }

}

void key_points(cv::Mat& img, const std::map<int, KeyPoint>& keypoints) {
    for (const auto& keyP : keypoints) {
        cv::circle(img, cv::Point(keyP.second.x, keyP.second.y), 2, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }
}

namespace Pose {
    class ModelImpl : public Model {
    public:
        error_e Load(const std::string& model_path);

        virtual error_e Run(const cv::Mat &frame, std::vector<yolov8_result>& det_results) override;
    private:
        error_e preprocess(
            const cv::Mat& src_frame, 
            const std::vector<yolov8_result>& det_results, 
            std::vector<AffineImg>& input_imgs
        );
        error_e inference(InferenceData& output_data);
        error_e postprocess(const AffineImg& input_img, std::map<int, KeyPoint>& pose_points);

    private:
        std::shared_ptr<ACEngine>   engine_;

        ac_engine_attr              input_attr_;
        ac_engine_attrs             output_attrs_;
    };

    error_e ModelImpl::Load(const std::string& model_path) {
        engine_ = create_engine(model_path, false);
        if (!engine_){
            LOG_ERROR("Deserialize engine failed.");
            return LOAD_MODEL_FAIL;
        }

        engine_->Print();
        input_attr_   = engine_->GetInputAttrs()[0];
        output_attrs_ = engine_->GetOutputAttrs();
        
        return SUCCESS;
    }

    error_e ModelImpl::preprocess(
        const cv::Mat& src_frame, 
        const std::vector<yolov8_result>& det_results, 
        std::vector<AffineImg>& input_imgs
    ) {
        int input_w     = input_attr_.dims[3];
        int input_h     = input_attr_.dims[2];

        for (size_t i = 0; i < det_results.size(); i++) {

            cv::Mat pose_img = src_frame(det_results[i].box).clone();
            cv::Mat pose_rgb, pose_resized, normalized_img;
            cv::cvtColor(pose_img, pose_rgb, cv::COLOR_BGR2RGB);
            cv::resize(pose_rgb, pose_resized, cv::Size(input_w, input_h));
            
            pose_resized.convertTo(normalized_img, CV_32F);
            cv::Scalar mean(123.675, 116.28, 103.53);
            cv::Scalar std(58.395, 57.12, 57.375);
            normalized_img = (normalized_img - mean) / std;

            AffineImg affine;
            affine.src_img = pose_img;
            affine.norm_img = normalized_img;

            input_imgs.push_back(affine);
        }

        return SUCCESS;
    }

    error_e ModelImpl::inference(InferenceData& infer_output_data) {
        engine_->GetInferOutput(infer_output_data);
        return SUCCESS;
    }

    error_e ModelImpl::postprocess(const AffineImg& input_img, std::map<int, KeyPoint>& pose_points) {
        int input_w     = input_attr_.dims[3];
        int input_h     = input_attr_.dims[2];
        int image_w     = input_img.src_img.cols;
        int image_h     = input_img.src_img.rows;

        float radio_w = (float)image_w / (float)input_w;
        float radio_h = (float)image_h / (float)input_h;

        InferenceData infer_output_data;
        engine_->GetInferOutput(infer_output_data);

        size_t output_num = infer_output_data.size();

        std::vector<int> output_size(output_num);
        for (size_t i = 0; i < output_num; ++i) {
            output_size[i] = infer_output_data[i].second;
        }

        void* output_data[output_num];
        for (int i = 0; i < output_num; i++) {
            output_data[i] = (void *)infer_output_data[i].first;
        }

        pose_decode((float *)output_data[0], (float *)output_data[1], pose_points);

        for (auto& point : pose_points) {
            point.second.x *= radio_w;
            point.second.y *= radio_h;
        }

        return SUCCESS;
    }

    error_e ModelImpl::Run(const cv::Mat& frame, std::vector<yolov8_result>& det_results) {
        std::vector<AffineImg> input_imgs;
        auto ret = preprocess(frame, det_results, input_imgs);
        if (ret != SUCCESS){
            LOG_ERROR("pose preprocess fail ...");
            return ret;
        }

        for (size_t i = 0; i < input_imgs.size(); i++) {
            InferenceData infer_input_data;
            infer_input_data.emplace_back(
                std::make_pair((void *)input_imgs[i].norm_img.ptr<float>(), input_imgs[i].norm_img.total() * input_imgs[i].norm_img.elemSize())
            );
            engine_->BindingInput(infer_input_data);

            std::map<int, KeyPoint> item_points;
            postprocess(input_imgs[i], item_points);

            // det_results[i].keypoints = item_points;

            key_points(input_imgs[i].src_img, item_points);

            char save_file[100];
            sprintf(save_file, "./output/pose/kkk/item%d.jpg", i);
            cv::imwrite(save_file, input_imgs[i].src_img);
        }
        return SUCCESS;
    }

    std::shared_ptr<Model> CreateInferModel(
        const std::string& pose_model_path,
        bool use_plugin
    ) {
        std::shared_ptr<ModelImpl> Instance(new ModelImpl());
        if (Instance->Load(pose_model_path) != SUCCESS) {
            Instance.reset();
        }
        return Instance;
    }
} // namespace Pose