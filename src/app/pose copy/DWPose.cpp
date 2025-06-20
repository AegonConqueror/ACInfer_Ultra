
#include "DWPose.h"

#include "engine.h"

#include "yolo/process/preprocess.h"

// 封装函数
std::vector<KeyPoint> get_keypoints_scores(
    const float* simcc_x,
    const float* simcc_y,
    int N, int K, int Wx, int Wy,
    float simcc_split_ratio = 1.0f)
{
    std::vector<KeyPoint> result(N * K);
    for (int i = 0; i < N * K; ++i) {
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

        result[i] = kp;
    }

    return result;
}

namespace DWPose {

    void letterbox_decode(std::vector<KeyPoint> &objects, bool hor, int pad) {
        for (auto &obj : objects) {
            if (hor) {
                obj.x -= pad;
                
            } else {
                obj.y -= pad;
            }
        }
    }
    
    class ModelImpl : public Model {
    public:
        error_e Load(const std::string &model_path, bool use_plugin);

        virtual error_e Run(const cv::Mat &frame, std::vector<KeyPoint>& results) override;
    private:
        error_e preprocess(const cv::Mat& src_frame, cv::Mat& dst_frame, cv::Mat& timg);
        error_e postprocess(std::vector<KeyPoint>& results);

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

    error_e ModelImpl::postprocess(std::vector<KeyPoint>& results) {
        InferenceData infer_output_data;
        engine_->GetInferOutput(infer_output_data);

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

        results = get_keypoints_scores((float *)output_data[0], (float *)output_data[1], N, K, Wx, Wy);

        return SUCCESS;
    }

    error_e ModelImpl::Run(const cv::Mat& frame, std::vector<KeyPoint>& results) {
        
        InferenceData infer_input_data;
        infer_input_data.emplace_back(
            std::make_pair((void *)frame.ptr<float>(), frame.total() * frame.elemSize())
        );
        engine_->BindingInput(infer_input_data);
        
        auto ret = postprocess(results);

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
