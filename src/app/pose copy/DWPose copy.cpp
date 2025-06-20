
#include "DWPose.h"

#include "engine.h"

#include "yolo/process/preprocess.h"

struct KeypointsAndScores {
    std::vector<float> keypoints; // (N, K, 2)
    std::vector<float> scores;    // (N, K)
};

// 封装函数
KeypointsAndScores get_keypoints_scores(
    const float* simcc_x,
    const float* simcc_y,
    int N, int K, int Wx, int Wy,
    float simcc_split_ratio = 1.0f)
{
    KeypointsAndScores result;
    result.keypoints.resize(N * K * 2, -1.0f);
    result.scores.resize(N * K, 0.0f);

    for (int i = 0; i < N * K; ++i) {
        const float* px = simcc_x + i * Wx;
        const float* py = simcc_y + i * Wy;

        // 最大值及索引
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

        int loc_offset = i * 2;
        if (val > 0.0f) {
            result.keypoints[loc_offset]     = static_cast<float>(max_x_idx);
            result.keypoints[loc_offset + 1] = static_cast<float>(max_y_idx);
        } else {
            result.keypoints[loc_offset]     = -1.0f;
            result.keypoints[loc_offset + 1] = -1.0f;
        }

        result.scores[i] = val;
    }

    // 如果需要除以 split_ratio
    if (std::abs(simcc_split_ratio - 1.0f) > 1e-6) {
        for (size_t i = 0; i < result.keypoints.size(); ++i) {
            result.keypoints[i] /= simcc_split_ratio;
        }
    }

    return result;
}

namespace DWPose {

    void letterbox_decode(KeypointsAndScores &objects, bool hor, int pad) {
        for (auto &obj : objects) {
            if (hor) {
                obj.box.x -= pad;

                if (task_type == TaskType::YOLOv8_POSE) {
                    for (auto &keypoint_item : obj.keypoints) {
                        keypoint_item.second.x -= pad;
                    }
                }
                
            } else {
                obj.box.y -= pad;

                if (task_type == TaskType::YOLOv8_POSE) {
                    for (auto &keypoint_item : obj.keypoints) {
                        keypoint_item.second.y -= pad;
                    }
                }
            }
        }
    }
    
    class ModelImpl : public Model {
    public:
        error_e Load(const std::string &model_path, bool use_plugin);

        virtual error_e Run(const cv::Mat &frame) override;
    private:
        error_e preprocess(const cv::Mat& src_frame, cv::Mat& dst_frame, cv::Mat& timg);
        error_e inference(InferenceData &output_data);
        error_e postprocess(const cv::Mat &frame, InferenceData &output_data);

    private:
        std::shared_ptr<ACEngine>   engine_;

        ac_engine_attr              input_attr_;
        ac_engine_attrs             output_attrs_;

        LetterBoxInfo               letterbox_info_;
    };

    error_e ModelImpl::Load(const std::string &model_path, bool use_plugin) {
        engine_ = create_engine(model_path, use_plugin);
        if (!engine_){
            LOG_ERROR("Deserialize engine failed.");
            return LOAD_MODEL_FAIL;
        }

        engine_->Print();
        input_attr_    = engine_->GetInputAttrs()[0];
        output_attrs_ = engine_->GetOutputAttrs();
        return SUCCESS;
    }

    error_e ModelImpl::preprocess(const cv::Mat& src_frame, cv::Mat& dst_frame, cv::Mat& timg) {
        int input_w     = input_attr_.dims[3];
        int input_h     = input_attr_.dims[2];
        
        float wh_ratio = (float)input_w / (float)input_h;
        letterbox_info_ = letterbox(src_frame, dst_frame, wh_ratio);

        cv::Mat rgb_img, resize_img;
        cv::cvtColor(dst_frame, rgb_img, cv::COLOR_BGR2RGB);
        cv::resize(rgb_img, resize_img, cv::Size(input_w, input_h));
        cv::dnn::blobFromImage(resize_img, timg, 1.0 / 255, cv::Size(input_w, input_h));
        timg.convertTo(timg, CV_32F);

        return SUCCESS;
    }

    error_e ModelImpl::inference(InferenceData &infer_output_data) {
        engine_->GetInferOutput(infer_output_data);
        return SUCCESS;
    }

    error_e ModelImpl::postprocess(const cv::Mat &frame, InferenceData &infer_output_data) {
        size_t output_num = infer_output_data.size();

        std::vector<int> output_size(output_num);
        for (size_t i = 0; i < output_num; ++i) {
            LOG_INFO("ouput %d size %d", i, infer_output_data[i].second);
            output_size[i] = infer_output_data[i].second;
        }

        void* output_data[output_num];
        for (int i = 0; i < output_num; i++) {
            output_data[i] = (void *)infer_output_data[i].first;
        }

        int N = 1;
        int K = 133;
        int Wx = 576;
        int Wy = 786;

        float simcc_split_ratio = 2.0;

        KeypointsAndScores res = get_keypoints_scores((float *)output_data[0], (float *)output_data[1], N, K, Wx, Wy);

        // 打印前几个结果
        for (int i = 0; i < std::min(5, N * K); ++i) {
            std::cout << "Loc: (" << res.keypoints[i * 2] << ", " << res.keypoints[i * 2 + 1]
                    << "), Val: " << res.scores[i] << std::endl;
        }

        return SUCCESS;
    }

    error_e ModelImpl::Run(const cv::Mat &frame) {

        cv::Mat letterbox_frame, timg;
        auto ret = preprocess(frame, letterbox_frame, timg);
        if (ret != SUCCESS){
            LOG_ERROR("yolov8 preprocess fail ...");
            return ret;
        }
        
        InferenceData infer_input_data;
        infer_input_data.emplace_back(
            std::make_pair((void *)timg.ptr<float>(), timg.total() * timg.elemSize())
        );
        engine_->BindingInput(infer_input_data);

        InferenceData infer_output_data;
        ret = inference(infer_output_data);
        if (ret != SUCCESS){
            LOG_ERROR("yolov8 inference fail ...");
            return ret;
        }
        
        ret = postprocess(letterbox_frame, infer_output_data);
        if (ret != SUCCESS){
            LOG_ERROR("yolov8 postprocess fail ...");
            return ret;
        }

        return SUCCESS;
    }

    std::shared_ptr<Model> CreateInferModel(
        const std::string &model_path,
        bool use_plugin
    ) {
        std::shared_ptr<ModelImpl> Instance(new ModelImpl());
        if (Instance->Load(model_path, use_plugin) != SUCCESS) {
            Instance.reset();
        }
        return Instance;
    }

} // namespace DWPose
