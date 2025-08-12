#include "pose.h"

#include "engine.h"

#include <unordered_set>

typedef struct AffineImg {
    cv::Mat src_img;
    cv::Mat timg;
    cv::Point2f center;
    cv::Point2f scale;
} AffineImg;

// std::unordered_set<int> choose_index{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 83, 85, 87, 89};
std::unordered_set<int> choose_index{0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 83, 85, 87, 89};

void dw_postprocess(
    const float* simcc_x,
    const float* simcc_y,
    std::map<int, KeyPoint>& pose_points,
    float simcc_split_ratio = 2.0f
) {
    int N = 1;
    int K = 133;
    int Wx = 384; // 576
    int Wy = 512; // 768

    int index = 0;
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            if (!choose_index.count(k))
                continue;
            
            int idx = n * K + k;

            int max_x_index = 0;
            float max_x_val = simcc_x[idx * Wx];

            for (int i = 1; i < Wx; ++i) {
                float val = simcc_x[idx * Wx + i];
                if (val > max_x_val) {
                    max_x_val = val;
                    max_x_index = i;
                }
            }

            int max_y_index = 0;
            float max_y_val = simcc_y[idx * Wy];
            for (int i = 1; i < Wy; ++i) {
                float val = simcc_y[idx * Wy + i];
                if (val > max_y_val) {
                    max_y_val = val;
                    max_y_index = i;
                }
            }

            float confidence = std::min(max_x_val, max_y_val);
            KeyPoint kp;
            kp.score = confidence;
            if (confidence > 0.0f) {
                kp.x = static_cast<float>(max_x_index) / simcc_split_ratio;
                kp.y = static_cast<float>(max_y_index) / simcc_split_ratio;
            } else {
                kp.x = -1.0f / simcc_split_ratio;
                kp.y = -1.0f / simcc_split_ratio;
            }

            kp.id = index;
            pose_points.insert({index, kp});
            index++;
        }
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

void padding_roi_img(
    const cv::Mat& src_frame, 
    const cv::Rect& roi, 
    cv::Mat& roi_img, 
    cv::Rect2f& padding_roi,
    float padding = 1.25f
) {
    padding_roi.x  = (float)roi.x - roi.width * (padding - 1.0) * 0.5;
    padding_roi.y  = (float)roi.y - roi.height * (padding - 1.0) * 0.5;
    padding_roi.width = roi.width * padding;
    padding_roi.height = roi.height * padding;

    padding_roi = padding_roi & cv::Rect2f(0, 0, src_frame.cols, src_frame.rows);

    roi_img = src_frame(padding_roi).clone();   
}

namespace Pose {
    class ModelImpl : public Model {
    public:
        error_e Load(const std::string& model_path);

        virtual error_e Run(const cv::Mat& frame, std::vector<yolo_result>& det_results) override;
    private:
        error_e preprocess(const cv::Mat& src_frame, const std::vector<yolo_result>& dets, std::vector<AffineImg>& input_imgs);
        error_e inference(InferenceData& output_data);
        error_e postprocess(const AffineImg& affine, KeyPoints& keypoints);

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

    error_e ModelImpl::preprocess(const cv::Mat& src_frame, const std::vector<yolo_result>& dets, std::vector<AffineImg>& input_imgs) {
        int input_w     = input_attr_.dims[3];
        int input_h     = input_attr_.dims[2];
        int image_w     = src_frame.cols;
        int image_h     = src_frame.rows;

        for (size_t i = 0; i < dets.size(); i++) {

            auto det_box = dets[i].box;
            cv::Rect box{det_box.left, det_box.top, det_box.right - det_box.left, det_box.bottom - det_box.top};

            cv::Point2f center;
            cv::Point2f scale;
            bbox_xyxy2cs(box, center, scale);

            cv::Mat resized_img, rgb_img, normalized_img;

            resized_img = top_down_affine(input_w, input_h, scale, center, src_frame);
            cv::cvtColor(resized_img, rgb_img, cv::COLOR_BGR2RGB);

            rgb_img.convertTo(normalized_img, CV_32F);
            cv::Scalar mean(123.675, 116.28, 103.53);
            cv::Scalar std(58.395, 57.12, 57.375);
            normalized_img = (normalized_img - mean) / std;

            AffineImg affine;
            affine.src_img = resized_img;
            affine.timg = normalized_img;
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

    error_e ModelImpl::postprocess(const AffineImg& affine, KeyPoints& keypoints) {
        int input_w     = input_attr_.dims[3];
        int input_h     = input_attr_.dims[2];
        int image_w     = affine.src_img.cols;
        int image_h     = affine.src_img.rows;

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

        dw_postprocess((float *)output_data[0], (float *)output_data[1], keypoints);

        for (auto& point : keypoints) {
            point.second.x = point.second.x / input_w * affine.scale.x + affine.center.x - affine.scale.x / 2.0f;
            point.second.y = point.second.y / input_h * affine.scale.y + affine.center.y - affine.scale.y / 2.0f;
        }

        return SUCCESS;
    }

    error_e ModelImpl::Run(const cv::Mat& frame, std::vector<yolo_result>& det_results) {
        int input_w     = input_attr_.dims[3];
        int input_h     = input_attr_.dims[2];
        int image_w     = frame.cols;
        int image_h     = frame.rows;

        cv::Mat testMat = frame.clone();

        auto time_start = iTime::timestamp_now();

        std::vector<AffineImg> input_imgs;
        auto ret = preprocess(frame, det_results, input_imgs);
        if (ret != SUCCESS){
            LOG_ERROR("pose preprocess fail ...");
            return ret;
        }

        for (size_t i = 0; i < input_imgs.size(); i++) {
            InferenceData infer_input_data;

            int H = input_h;
            int W = input_w;
            int C = 3;

            std::vector<float> input_tensor_values(C * H * W); 

            // 将 HWC → CHW
            for (int c = 0; c < C; ++c) {
                for (int h = 0; h < H; ++h) {
                    for (int w = 0; w < W; ++w) {
                        input_tensor_values[c * H * W + h * W + w] =
                            input_imgs[i].timg.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
            
            infer_input_data.emplace_back(
                std::make_pair((void *)input_tensor_values.data(), input_imgs[i].timg.total() * input_imgs[i].timg.elemSize())
            );

            engine_->BindingInput(infer_input_data);

            KeyPoints item_points;
            postprocess(input_imgs[i], item_points);

            auto i_det_box = det_results[i].box;

            for (auto& point : item_points) {
                if (
                    point.second.x < i_det_box.left || 
                    point.second.y < i_det_box.top || 
                    point.second.x > i_det_box.right ||
                    point.second.y > i_det_box.bottom
                ) {
                    point.second.x = 0;
                    point.second.y = 0;
                    point.second.score = 0;
                }
            }

            det_results[i].keypoints = item_points;
        }
        
        auto time_end= iTime::timestamp_now();
        LOG_INFO("inference done %lld ms !", time_end - time_start);
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