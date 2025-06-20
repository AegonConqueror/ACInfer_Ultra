
#include "yolov8.h"

#include "engine.h"
#include "process/preprocess.h"
#include "process/yolov8_postprocess.h"

void yolov8_pose_decode_gpu(
    float* output_data, int* output_size, int output_num,
    int* num_dets, int* det_classes, float* det_scores, float* det_boxes, float* det_keypoints,
    int input_w, int input_h, int image_w, int image_h, int class_num,
    float conf_thres, float nms_thres, int keypoint_num
);

namespace YOLOv8 {

    void letterbox_decode(std::vector<yolov8_result>& objects, bool hor, int pad, TaskType task_type) {
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
        error_e Load(const std::string& model_path, const TaskType task_type, bool use_plugin);

        virtual error_e Run(const cv::Mat& frame, std::vector<yolov8_result>& objects) override;
    private:
        error_e preprocess(const cv::Mat& src_frame, cv::Mat& dst_frame, cv::Mat& timg);
        error_e inference(InferenceData& output_data);
        error_e postprocess(const cv::Mat& frame, InferenceData& output_data, std::vector<yolov8_result>& objects);

    private:
        std::shared_ptr<ACEngine>   engine_;
        TaskType                    task_type_;
        LetterBoxInfo               letterbox_info_;

        ac_engine_attr              input_attr_;
        ac_engine_attrs             output_attrs_;

        int                         model_class_num_;
        int                         model_keypoints_num_;
    };

    error_e ModelImpl::Load(const std::string& model_path, const TaskType task_type, bool use_plugin) {
        task_type_ = task_type;
        engine_ = create_engine(model_path, use_plugin);
        if (!engine_){
            LOG_ERROR("Deserialize engine failed.");
            return LOAD_MODEL_FAIL;
        }

        engine_->Print();
        input_attr_    = engine_->GetInputAttrs()[0];
        output_attrs_ = engine_->GetOutputAttrs();

        int cls_index = engine_->GetOutputIndex("cls1");
        model_class_num_ = output_attrs_[cls_index].dims[1];

        if (task_type_ == TaskType::YOLOv8_POSE) {
            int ps_index = engine_->GetOutputIndex("ps1");
            model_keypoints_num_ = static_cast<int>(output_attrs_[ps_index].dims[1] / 3);
        }
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

    error_e ModelImpl::inference(InferenceData& infer_output_data) {
        engine_->GetInferOutput(infer_output_data);
        return SUCCESS;
    }

    error_e ModelImpl::postprocess(const cv::Mat& frame, InferenceData& infer_output_data, std::vector<yolov8_result>& objects) {
        int input_w     = input_attr_.dims[3];
        int input_h     = input_attr_.dims[2];
        int image_w     = frame.cols;
        int image_h     = frame.rows;

        size_t output_num = infer_output_data.size();
        void* output_data[output_num];

        std::vector<float> detectiont_rects;
        std::vector<std::map<int, KeyPoint>> pose_keypoints;

        if (task_type_ == TaskType::YOLOv8_POSE) {
            std::vector<std::string> pose_names = {"reg1", "cls1", "reg2", "cls2", "reg3", "cls3", "ps1", "ps2", "ps3"};
            if (pose_names.size() != output_num) return LOAD_MODEL_FAIL;

#if USE_TENSORRT == 1
            std::vector<int> output_size(output_num);
            for (size_t i = 0; i < output_num; ++i) {
                output_size[i] = infer_output_data[engine_->GetOutputIndex(pose_names[i])].second;
            }

            int total_size = std::accumulate(output_size.begin(), output_size.end(), 0);

            float* h_output_data = (float *)malloc(total_size * sizeof(float));

            int pos = 0;
            for (size_t i = 0; i < output_num; ++i) {
                float* src = (float *)infer_output_data[engine_->GetOutputIndex(pose_names[i])].first;
                size_t offset = output_size[i];

                memcpy(h_output_data, src, offset * sizeof(float));
                pos += offset;
            }

            LOG_INFO("using gpu kernel decode pose");

            int keepTopK = 20;
            
            int num_dets[1];
            int det_classes[1 * keepTopK]; 
            float det_scores[1 * keepTopK]; 
            float det_boxes[1 * keepTopK];
            float det_keypoints[1 * keepTopK * 3 * model_keypoints_num_ ];

            yolov8_pose_decode_gpu(
                h_output_data, output_size.data(), output_num, 
                num_dets, det_classes, det_scores, det_boxes, det_keypoints,
                input_w, input_h, image_w, image_h, model_class_num_,
                0.45, 0.45, model_keypoints_num_
            );

            for (int i = 0; i < num_dets[1]; i++) {
                yolov8_result dr;

                int classId = det_classes[i + keepTopK];
                float conf = det_scores[i + keepTopK];
                
                int xmin = int(det_boxes[i * 4 + 0 + keepTopK * 4]);
                int ymin = int(det_boxes[i * 4 + 1 + keepTopK * 4]);
                int xmax = int(det_boxes[i * 4 + 2 + keepTopK * 4]);
                int ymax = int(det_boxes[i * 4 + 3 + keepTopK * 4]);

                std::map<int, KeyPoint> kp_map;
                for (int k = 0; k < model_keypoints_num_; k++) {
                    KeyPoint kp;
                    kp.x = det_keypoints[k * model_keypoints_num_ * 3 + i * 3 + 0 + keepTopK * 3 * model_keypoints_num_];
                    kp.y = det_keypoints[k * model_keypoints_num_ * 3 + i * 3 + 1 + keepTopK * 3 * model_keypoints_num_];
                    kp.score = det_keypoints[k * model_keypoints_num_ * 3 + i * 3 + 2 + keepTopK * 3 * model_keypoints_num_];
                    kp.id = k;
                    kp_map[k] = kp;
                }

                dr.class_id = classId;
                dr.confidence = conf;
                dr.box = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
                dr.keypoints = kp_map;

                objects.push_back(dr);
            }

            free(h_output_data);

            return SUCCESS;
#else       

            for (size_t i = 0; i < output_num; i++) {
                output_data[i] = (void *)infer_output_data[engine_->GetOutputIndex(pose_names[i])].first;
            }

            yolov8::PostprocessSplit_POSE(
                (float **)output_data, detectiont_rects, pose_keypoints, 
                input_w, input_h, model_class_num_, model_keypoints_num_
            );

            for (auto &kp : pose_keypoints) {
                for (auto &kp_item : kp) {
                    kp_item.second.x = kp_item.second.x * float(image_w);
                    kp_item.second.y = kp_item.second.y * float(image_h);
                }
            }
#endif
        }

        for (int i = 0; i < detectiont_rects.size(); i += 6) {
            int classId = int(detectiont_rects[i + 0]);
            float conf = detectiont_rects[i + 1];
            int xmin = int(detectiont_rects[i + 2] * float(image_w) + 0.5);
            int ymin = int(detectiont_rects[i + 3] * float(image_h) + 0.5);
            int xmax = int(detectiont_rects[i + 4] * float(image_w) + 0.5);
            int ymax = int(detectiont_rects[i + 5] * float(image_h) + 0.5);

            yolov8_result dr;
            dr.box = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
            dr.class_id = classId;
            dr.confidence = conf;
            if (task_type_ == TaskType::YOLOv8_POSE) {
                dr.keypoints = pose_keypoints[static_cast<int>(i / 6)];
            }
            objects.push_back(dr);
        }
        return SUCCESS;
    }

    error_e ModelImpl::Run(const cv::Mat& frame, std::vector<yolov8_result>& objects) {

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

        ret = postprocess(letterbox_frame, infer_output_data, objects);
        if (ret != SUCCESS){
            LOG_ERROR("yolov8 postprocess fail ...");
            return ret;
        }   

        letterbox_decode(objects, letterbox_info_.hor, letterbox_info_.pad, task_type_);
        return SUCCESS;
    }

    std::shared_ptr<Model> CreateInferModel(const std::string& model_path, const TaskType task_type, bool use_plugin) {
        std::shared_ptr<ModelImpl> Instance(new ModelImpl());
        if (Instance->Load(model_path, task_type, use_plugin) != SUCCESS) {
            Instance.reset();
        }
        return Instance;
    }
} // namespace YOLOv8
