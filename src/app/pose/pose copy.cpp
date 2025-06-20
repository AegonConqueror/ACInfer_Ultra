#include "pose.h"

#include "engine.h"

typedef struct AffineImg {
    cv::Mat img;
    cv::Point2f center;
    cv::Point2f scale;
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

cv::Point2f fix_aspect_ratio(const cv::Point2f& scale, float aspect_ratio) {
    float w = scale.x;
    float h = scale.y;
    if (w > aspect_ratio * h) {
        h = w / aspect_ratio;
    } else if (w < aspect_ratio * h) {
        w = h * aspect_ratio;
    }
    return cv::Point2f(w, h);
}

cv::Mat get_warp_matrix(
    const cv::Point2f& center, 
    const cv::Point2f& scale, 
    float rot, const int input_w, const int input_h
) {
    float rot_rad = rot * CV_PI / 180.0;

    // 分别定义源点和目标点
    float src_w = scale.x;
    float src_h = scale.y;

    // 方向向量
    cv::Point2f src_dir = cv::Point2f(0.0, -0.5 * src_w);
    cv::Point2f dst_dir = cv::Point2f(0.0, -0.5 * input_w);

    // 旋转
    float sn = sin(rot_rad);
    float cs = cos(rot_rad);

    cv::Point2f src_rotated = cv::Point2f(
        src_dir.x * cs - src_dir.y * sn,
        src_dir.x * sn + src_dir.y * cs
    );

    cv::Point2f dst_rotated = dst_dir; // 目标方向固定

    // 三个点确定仿射
    std::vector<cv::Point2f> src_pts(3);
    std::vector<cv::Point2f> dst_pts(3);

    src_pts[0] = center;
    src_pts[1] = center + src_rotated;
    // 第三个点是求垂直方向
    cv::Point2f src_third = center + cv::Point2f(-src_rotated.y, src_rotated.x);
    src_pts[2] = src_third;

    cv::Point2f dst_center = cv::Point2f(input_w / 2.0, input_h / 2.0);
    dst_pts[0] = dst_center;
    dst_pts[1] = dst_center + dst_rotated;
    dst_pts[2] = dst_center + cv::Point2f(-dst_rotated.y, dst_rotated.x);

    cv::Mat warp_mat = cv::getAffineTransform(src_pts, dst_pts);
    return warp_mat;
}

void bbox_xyxy2cs(const cv::Rect& bbox, cv::Point2f& center, cv::Point2f& scale, float padding = 1.25f) {
    int x1 = bbox.x;
    int y1 = bbox.y;
    int x2 = bbox.x + bbox.width;
    int y2 = bbox.y + bbox.height;

    float cx = (x1 + x2) * 0.5f;
    float cy = (y1 + y2) * 0.5f;
    float w = (x2 - x1) * padding;
    float h = (y2 - y1) * padding;

    center = cv::Point2f(cx, cy);
    scale = cv::Point2f(w, h);
}

cv::Mat top_down_affine(
    const int input_w, const int input_h,
    cv::Point2f& bbox_scale,
    const cv::Point2f& bbox_center,
    const cv::Mat& img
) {

    cv::Size input_size;
    input_size.width = input_w;
    input_size.height = input_h;

    bbox_scale = fix_aspect_ratio(bbox_scale, (float)input_w / (float)input_h);
    cv::Mat warp_mat = get_warp_matrix(bbox_center, bbox_scale, 0, input_w, input_h);

    cv::Mat warped_img;
    cv::warpAffine(img, warped_img, warp_mat, input_size, cv::INTER_LINEAR);

    return warped_img;
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
        error_e postprocess(const AffineImg& affine, std::map<int, KeyPoint>& pose_points);

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
            cv::Point2f center;
            cv::Point2f scale;
            bbox_xyxy2cs(det_results[i].box, center, scale);

            cv::Mat resized_img = top_down_affine(input_w, input_h, scale, center, src_frame);

            cv::Mat normalized_img;
            resized_img.convertTo(normalized_img, CV_32F);
            cv::Scalar mean(123.675, 116.28, 103.53);
            cv::Scalar std(58.395, 57.12, 57.375);
            normalized_img = (normalized_img - mean) / std;

            AffineImg affine;
            affine.img = normalized_img;
            affine.center = center;
            affine.scale = scale;
            input_imgs.push_back(affine);
        }

        return SUCCESS;
    }

    error_e ModelImpl::inference(InferenceData& infer_output_data) {
        engine_->GetInferOutput(infer_output_data);
        return SUCCESS;
    }

    error_e ModelImpl::postprocess(const AffineImg& affine, std::map<int, KeyPoint>& pose_points) {
        int input_w     = input_attr_.dims[3];
        int input_h     = input_attr_.dims[2];

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
            point.second.x /= input_w * affine.scale.x + affine.center.x - affine.scale.x / 2.0f;
            point.second.y /= input_h * affine.scale.y + affine.center.y - affine.scale.y / 2.0f;
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
                std::make_pair((void *)input_imgs[i].img.ptr<float>(), input_imgs[i].img.total() * input_imgs[i].img.elemSize())
            );
            engine_->BindingInput(infer_input_data);

            std::map<int, KeyPoint> item_points;
            postprocess(input_imgs[i], item_points);

            det_results[i].keypoints = item_points;
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