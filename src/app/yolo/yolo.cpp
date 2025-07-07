#include "yolo.h"

#include "engine.h"

#include "process/preprocess.h"
#include "process/yolov8_postprocess.h"

#include "process/yolov8_postprocess.cuh"


namespace YOLO {

    void letterbox_decode(std::vector<yolo_result>& objects, bool hor, int pad, Task task_type) {
        for (auto& obj : objects) {
            if (hor) {
                obj.box.left -= pad;
                obj.box.right -= pad;

                if (task_type == Task::YOLO_POSE) {
                    for (auto &keypoint_item : obj.keypoints) {
                        keypoint_item.second.x -= pad;
                    }
                }
                
            } else {
                obj.box.bottom -= pad;
                obj.box.top -= pad;

                if (task_type == Task::YOLO_POSE) {
                    for (auto &keypoint_item : obj.keypoints) {
                        keypoint_item.second.y -= pad;
                    }
                }
            }
        }
    }

    class ModelImpl : public Model {
    public:
        error_e Load(
            const std::string& model_path, 
            const Type yolo_type, 
            const Task task_type,
            const DecodeMethod decode_type, 
            bool use_plugin
        );

        virtual error_e Run(const cv::Mat& frame, YoloResults& resluts) override;
    private:
        error_e preprocess(const cv::Mat& src_frame, cv::Mat& dst_frame, cv::Mat& timg);
        error_e inference(InferenceData& output_data);
        error_e postprocess(const cv::Mat& src_frame, InferenceData& infer_output_data, YoloResults& resluts);

    private:
        std::shared_ptr<ACEngine>   engine_;

        Type                        yolo_type_;
        Task                        task_type_;
        DecodeMethod                decode_type_;

        LetterBoxInfo               letterbox_info_;

        ac_engine_attr              input_attr_;
        ac_engine_attrs             output_attrs_;

        int                         model_class_num_;
        int                         model_keypoints_num_;
    };

    error_e ModelImpl::Load(
        const std::string& model_path, 
        const Type yolo_type, 
        const Task task_type,
        const DecodeMethod decode_type, 
        bool use_plugin
    ) {
        yolo_type_ = yolo_type;
        task_type_ = task_type;
        decode_type_ = decode_type;

        engine_ = create_engine(model_path, use_plugin);
        if (!engine_){
            LOG_ERROR("Deserialize engine failed.");
            return LOAD_MODEL_FAIL;
        }

        engine_->Print();
        input_attr_    = engine_->GetInputAttrs()[0];
        output_attrs_ = engine_->GetOutputAttrs();

        if (decode_type_ == DecodeMethod::CPU) {
            model_class_num_ = output_attrs_[engine_->GetOutputIndex("cls1")].dims[1];

            if (task_type_ == Task::YOLO_POSE) {
                model_keypoints_num_ = static_cast<int>(output_attrs_[engine_->GetOutputIndex("ps1")].dims[1] / 3);
            }
        }
        return SUCCESS;
    }

    error_e ModelImpl::preprocess(const cv::Mat& src_frame, cv::Mat& dst_frame, cv::Mat& timg) {
        int input_w     = input_attr_.dims[3];
        int input_h     = input_attr_.dims[2];
        
        float wh_ratio = (float)input_w / (float)input_h;
        letterbox_info_ = letterbox(src_frame, dst_frame, wh_ratio);

        if (yolo_type_ == Type::V8) {
            cv::Mat rgb_img, resize_img;
            cv::cvtColor(dst_frame, rgb_img, cv::COLOR_BGR2RGB);
            cv::resize(rgb_img, resize_img, cv::Size(input_w, input_h));
            cv::dnn::blobFromImage(resize_img, timg, 1.0 / 255, cv::Size(input_w, input_h));
            timg.convertTo(timg, CV_32F);
        }
        else if (yolo_type_ == Type::X) {
            cv::Mat rgb_img;
            cv::cvtColor(dst_frame, rgb_img, cv::COLOR_BGR2RGB);
            cv::resize(rgb_img, timg, cv::Size(input_w, input_h));

            timg.convertTo(timg, CV_32F);
        }
        else {
            LOG_ERROR("Unsupport type %d", yolo_type_);
        }
        return SUCCESS;
    }

    error_e ModelImpl::inference(InferenceData& infer_output_data) {
        engine_->GetInferOutput(infer_output_data);
        return SUCCESS;
    }

    error_e ModelImpl::postprocess(const cv::Mat& src_frame, InferenceData& infer_output_data, YoloResults& resluts) {
        int input_w = input_attr_.dims[3];
        int input_h = input_attr_.dims[2];
        int image_w = src_frame.cols;
        int image_h = src_frame.rows;

        size_t output_num = infer_output_data.size();

        if (decode_type_ == DecodeMethod::CPU) {
            void* output_data[output_num];

            std::vector<float> detectiont_rects;
            KeyPointsArray pose_keypoints;

            if (yolo_type_ == Type::V8) {
                if (task_type_ == Task::YOLO_POSE) {
                    std::vector<std::string> pose_names = {"reg1", "cls1", "reg2", "cls2", "reg3", "cls3", "ps1", "ps2", "ps3"};
                    if (pose_names.size() != output_num) return LOAD_MODEL_FAIL;

                    for (size_t i = 0; i < output_num; i++) {
                        output_data[i] = (void *)infer_output_data[engine_->GetOutputIndex(pose_names[i])].first;
                    }

                    yolov8::PostprocessSplit_POSE(
                        (float **)output_data, detectiont_rects, pose_keypoints, 
                        input_w, input_h, model_class_num_, model_keypoints_num_,
                        0.25
                    );

                    for (auto& kp : pose_keypoints) {
                        for (auto& kp_item : kp) {
                            kp_item.second.x = kp_item.second.x * float(image_w);
                            kp_item.second.y = kp_item.second.y * float(image_h);
                        }
                    }
                }

                for (int i = 0; i < detectiont_rects.size(); i += 6) {
                    int classId = int(detectiont_rects[i + 0]);
                    float conf = detectiont_rects[i + 1];
                    int xmin = int(detectiont_rects[i + 2] * float(image_w) + 0.5);
                    int ymin = int(detectiont_rects[i + 3] * float(image_h) + 0.5);
                    int xmax = int(detectiont_rects[i + 4] * float(image_w) + 0.5);
                    int ymax = int(detectiont_rects[i + 5] * float(image_h) + 0.5);

                    yolo_result dr;
                    dr.box = Box(xmin, ymin, xmax, ymax);
                    dr.classId = classId;
                    dr.score = conf;
                    if (task_type_ == Task::YOLO_POSE) {
                        dr.keypoints = pose_keypoints[static_cast<int>(i / 6)];
                    }
                    resluts.push_back(dr);
                }
            }
        } else {
            if (yolo_type_ == Type::V8) {
                if (task_type_ == Task::YOLO_POSE) {
                    std::vector<std::string> pose_names = {"reg1", "cls1", "reg2", "cls2", "reg3", "cls3", "ps1", "ps2", "ps3"};
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

                    LOG_INFO(">>> num_dets[1] %d", num_dets[1]);
                }
            }
        }
        
        return SUCCESS;
    }

    error_e ModelImpl::Run(const cv::Mat& frame, YoloResults& resluts) {
        auto time_0 = iTime::timestamp_now();

        cv::Mat letterbox_frame, timg;
        auto ret = preprocess(frame, letterbox_frame, timg);
        if (ret != SUCCESS){
            LOG_ERROR("yolo preprocess fail ...");
            return ret;
        }
        
        InferenceData infer_input_data;
        infer_input_data.emplace_back(
            std::make_pair((void *)timg.ptr<float>(), timg.total() * timg.elemSize())
        );
        engine_->BindingInput(infer_input_data);

        auto time_1 = iTime::timestamp_now();
        LOG_DEBUG("preprocess done %lld ms !", time_1 - time_0);

        InferenceData infer_output_data;
        ret = inference(infer_output_data);
        if (ret != SUCCESS){
            LOG_ERROR("yolov8 inference fail ...");
            return ret;
        }

        auto time_2 = iTime::timestamp_now();
        LOG_DEBUG("inference done %lld ms !", time_2 - time_1);

        ret = postprocess(letterbox_frame, infer_output_data, resluts);
        if (ret != SUCCESS){
            LOG_ERROR("yolov8 postprocess fail ...");
            return ret;
        }

        letterbox_decode(resluts, letterbox_info_.hor, letterbox_info_.pad, task_type_);

        auto time_3 = iTime::timestamp_now();
        LOG_DEBUG("postprocess done %lld ms !", time_3 - time_2);
        return SUCCESS;
    }

    std::shared_ptr<Model> CreateInferModel(
        const std::string& model_path,
        const Type yolo_type,
        const Task task_type,
        const DecodeMethod decode_type,
        bool use_plugin
    ) {
        std::shared_ptr<ModelImpl> Instance(new ModelImpl());
        if (Instance->Load(model_path, yolo_type, task_type, decode_type, use_plugin) != SUCCESS) {
            Instance.reset();
        }
        return Instance;
    }
    
} // namespace YOLO
