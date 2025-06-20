#include "pose.h"

#include "engine.h"

void key_points(cv::Mat& img, const std::map<int, KeyPoint>& keypoints) {
    int index = 0;
    for (const auto& keyP : keypoints) {
        index++;
        if (index >= 24 && index <= 90) 
            cv::circle(img, cv::Point(keyP.second.x, keyP.second.y), 2, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
        else
            continue;
    }
}

typedef struct AffineImg {
    cv::Mat src_img;
    cv::Mat img;
    cv::Point2f center;
    cv::Point2f scale;
} AffineImg;

void dw_postprocess(
    const float* simcc_x,
    const float* simcc_y,
    std::map<int, KeyPoint>& pose_points,
    float simcc_split_ratio = 2.0f
) {
    int N = 1;
    int K = 133;
    int Wx = 576;
    int Wy = 768;

    int test_max = 0;
    float max_x = simcc_x[0];

    for (int i = 1; i < K * Wx; ++i) {
        float v = simcc_x[i];
        if (v > max_x) {
            max_x = v;
            test_max = i;
        }
    }

    LOG_INFO("max val %f index %d", max_x, test_max);

    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            int idx = n * K + k;

            // 获取 simcc_x 的最大值及位置
            int max_x_index = 0;
            float max_x_val = simcc_x[idx * Wx];

            for (int i = 1; i < Wx; ++i) {
                float val = simcc_x[idx * Wx + i];
                if (val > max_x_val) {
                    max_x_val = val;
                    max_x_index = i;
                }
            }

            // 获取 simcc_y 的最大值及位置
            int max_y_index = 0;
            float max_y_val = simcc_y[idx * Wy];
            for (int i = 1; i < Wy; ++i) {
                float val = simcc_y[idx * Wy + i];
                if (val > max_y_val) {
                    max_y_val = val;
                    max_y_index = i;
                }
            }

            // 取较小值作为最终置信度
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

            kp.id = k;
            pose_points.insert({k, kp});
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
    // LOG_INFO("x1 %d, y1 %d, x2 %d, y2 %d", x1, y1, x2, y2);

    float cx = (x1 + x2) * 0.5f;
    float cy = (y1 + y2) * 0.5f;
    float w = (x2 - x1) * padding;
    float h = (y2 - y1) * padding;

    // LOG_INFO("cx %f, cy %f, w %f, h %f", cx, cy, w, h);

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

void process_keypoints(std::map<int, KeyPoint>& keypoints) {
    // Step 1: 计算 neck = 平均 5 和 6
    KeyPoint kp5 = keypoints[5];
    KeyPoint kp6 = keypoints[6];

    KeyPoint neck;
    neck.x = (kp5.x + kp6.x) / 2.0f;
    neck.y = (kp5.y + kp6.y) / 2.0f;

    // neck 的 score 判断
    bool valid_5 = kp5.score > 0.3f;
    bool valid_6 = kp6.score > 0.3f;
    neck.score = (valid_5 && valid_6) ? 1.0f : 0.0f;
    neck.id = 17;

    // Step 2: 插入为 id = 17
    keypoints[17] = neck;

    // Step 3: 根据映射关系重排 keypoints 到 new_map
    std::vector<int> mmpose_idx  = {17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3};
    std::vector<int> openpose_idx = {1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17};

    std::map<int, KeyPoint> new_keypoints = keypoints;  // 复制旧 keypoints

    for (size_t i = 0; i < mmpose_idx.size(); ++i) {
        int src = mmpose_idx[i];
        int dst = openpose_idx[i];
        if (keypoints.count(src)) {
            KeyPoint kp = keypoints[src];
            kp.id = dst;
            new_keypoints[dst] = kp;
        }
    }

    keypoints = new_keypoints;
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
        int image_w     = src_frame.cols;
        int image_h     = src_frame.rows;

        for (size_t i = 0; i < det_results.size(); i++) {
            auto box = det_results[i].box;

            cv::Point2f center;
            cv::Point2f scale;
            bbox_xyxy2cs(det_results[i].box, center, scale);

            cv::Mat resized_img = top_down_affine(input_w, input_h, scale, center, src_frame);

            cv::Mat rgb_img, normalized_img;
            cv::cvtColor(resized_img, rgb_img, cv::COLOR_BGR2RGB);

            // char save_file[100];
            // sprintf(save_file, "./output/pose/kkk/item%d.jpg", i);
            // cv::imwrite(save_file, rgb_img);

            // char open_file[100];
            // sprintf(open_file, "./python/DWPose/output0%d.jpg", i);

            // rgb_img = cv::imread(open_file);

            rgb_img.convertTo(normalized_img, CV_32F);
            cv::Scalar mean(123.675, 116.28, 103.53);
            cv::Scalar std(58.395, 57.12, 57.375);
            normalized_img = (normalized_img - mean) / std;

            // std::cout << "C++ normalized RGB values:\n";
            // int count = 0;
            // for (int h = 0; h < normalized_img.rows; ++h) {
            //     for (int w = 0; w < normalized_img.cols; ++w) {
            //         if (count >= 10) break;
            //         cv::Vec3f pixel = normalized_img.at<cv::Vec3f>(h, w);
            //         std::cout << "Pixel " << count << ": "
            //                 << "R=" << pixel[0] << ", "
            //                 << "G=" << pixel[1] << ", "
            //                 << "B=" << pixel[2] << std::endl;
            //         count++;
            //     }
            //     if (count >= 10) break;
            // }

            // sprintf(save_file, "./output/pose/kkk/norm_item%d.jpg", i);
            // cv::imwrite(save_file, normalized_img);

            AffineImg affine;
            affine.src_img = resized_img;
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

        dw_postprocess((float *)output_data[0], (float *)output_data[1], pose_points);

        for (auto& point : pose_points) {
            point.second.x = point.second.x / input_w * affine.scale.x + affine.center.x - affine.scale.x / 2.0f;
            point.second.y = point.second.y / input_h * affine.scale.y + affine.center.y - affine.scale.y / 2.0f;
        }

        // process_keypoints(pose_points);

        return SUCCESS;
    }

    error_e ModelImpl::Run(const cv::Mat& frame, std::vector<yolov8_result>& det_results) {
        int input_w     = input_attr_.dims[3];
        int input_h     = input_attr_.dims[2];
        int image_w     = frame.cols;
        int image_h     = frame.rows;

        cv::Mat testMat = frame.clone();

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
                            input_imgs[i].img.at<cv::Vec3f>(h, w)[c];
                    }
                }
            }
            
            infer_input_data.emplace_back(
                std::make_pair((void *)input_tensor_values.data(), input_imgs[i].img.total() * input_imgs[i].img.elemSize())
            );
            engine_->BindingInput(infer_input_data);

            std::map<int, KeyPoint> item_points;
            postprocess(input_imgs[i], item_points);

            // for (auto& keyP : item_points) {
            //     keyP.second.x /= (float)image_w;
            //     keyP.second.y /= (float)image_h;
            // }
            key_points(testMat, item_points);
            // char save_file[100];
            // sprintf(save_file, "./output/pose/kkk/item%d.jpg", i);
            // cv::imwrite(save_file, input_imgs[i].src_img);
        }

        //     char save_file[100];
        //     sprintf(save_file, "./output/pose/kkk/item%d.jpg", i);
            cv::imwrite("./output/test.jpg", testMat);

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