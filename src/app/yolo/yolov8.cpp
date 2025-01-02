
#include "yolov8.h"

#include "engine/engine.h"
#include "process/preprocess.h"
#include "process/yolov8_postprocess.h"

namespace YOLOv8 {

    void letterbox_decode(std::vector<yolov8_result> &objects, bool hor, int pad, TaskType task_type) {
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
        error_e Load(const std::string &model_path, const TaskType task_type, bool use_plugin);

        virtual error_e Run(const cv::Mat &frame, std::vector<yolov8_result> &objects) override;
    private:
        error_e preprocess(const cv::Mat &src_frame, cv::Mat &dst_frame, cv::Mat &timg);
        error_e postprocess(const cv::Mat &frame, std::vector<yolov8_result> &objects);

    private:
        std::shared_ptr<ACEngine>   engine_;
        TaskType                    task_type_;
        std::vector<int>            input_shape_;
        LetterBoxInfo               letterbox_info_;   
    };

    error_e ModelImpl::Load(const std::string &model_path, const TaskType task_type, bool use_plugin) {
        task_type_ = task_type;
        engine_ = create_engine(model_path, use_plugin);
        if (!engine_){
            LOG_ERROR("Deserialize engine failed.");
            return LOAD_MODEL_FAIL;
        }

        engine_->Print();
        input_shape_    = engine_->GetInputShape();
        return SUCCESS;
    }

    error_e ModelImpl::preprocess(const cv::Mat &src_frame, cv::Mat &dst_frame, cv::Mat &timg) {
        int input_w     = input_shape_[3];
        int input_h     = input_shape_[2];
        
        float wh_ratio = (float)input_w / (float)input_h;
        letterbox_info_ = letterbox(src_frame, dst_frame, wh_ratio);

        cv::Mat rgb_img, resize_img;
        cv::cvtColor(dst_frame, rgb_img, cv::COLOR_BGR2RGB);
        cv::resize(rgb_img, resize_img, cv::Size(input_w, input_h));
        cv::dnn::blobFromImage(resize_img, timg, 1.0 / 255, cv::Size(input_w, input_h));
        timg.convertTo(timg, CV_32F);
        
        return SUCCESS;
    }

    error_e ModelImpl::postprocess(const cv::Mat &frame, std::vector<yolov8_result> &objects) {
        int input_w     = input_shape_[3];
        int input_h     = input_shape_[2];
        int image_w     = frame.cols;
        int image_h     = frame.rows;

        InferenceDataType infer_output_data;
        engine_->GetInferOutput(infer_output_data);

        size_t output_num = infer_output_data.size();

        auto output_shapes = engine_->GetOutputShapes();
        auto class_num = output_shapes[1][1];

        void* output_data[output_num];

        for (size_t i = 0; i < output_num; i++) {
            output_data[i] = (void*)infer_output_data[i].first;
        }

        std::vector<float> detectiont_rects;
        std::vector<std::map<int, KeyPoint>> pose_keypoints;
        std::vector<cv::Mat> seg_masks;
        cv::Mat seg_mask = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC3);

        if (task_type_ == TaskType::YOLOv8_DET) {
            yolov8::PostprocessSplit_DET(
                (float **)output_data, detectiont_rects,
                input_w, input_h, class_num
            );
        }
        
        if (task_type_ == TaskType::YOLOv8_POSE) {
            yolov8::PostprocessSplit_POSE(
                (float **)output_data, detectiont_rects,
                pose_keypoints, input_w, input_h, class_num
            );

            for (auto &kp : pose_keypoints) {
                for (auto &kp_item : kp) {
                    kp_item.second.x = kp_item.second.x * float(image_w);
                    kp_item.second.y = kp_item.second.y * float(image_h);
                }
            }
        } 

        if (task_type_ == TaskType::YOLOv8_SEG) {
            yolov8::PostprocessSplit_SEG(
                (float **)output_data, detectiont_rects,
                seg_masks, input_w, input_h, class_num
            );
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
            
            if (task_type_ == TaskType::YOLOv8_SEG) {
                cv::Mat seg_cv = seg_masks[static_cast<int>(i / 6)];
                cv::Mat seg_cv_resize;
                cv::resize(seg_cv, seg_cv_resize, frame.size());
                cv::Mat roi;
                cv::inRange(seg_cv_resize(dr.box), cv::Scalar(0.5), cv::Scalar(1), roi);
                dr.mask = roi;
            }

            objects.push_back(dr);
        }

        return SUCCESS;
    }

    error_e ModelImpl::Run(const cv::Mat &frame, std::vector<yolov8_result> &objects) {
        cv::Mat letterbox_frame, timg;
        auto ret = preprocess(frame, letterbox_frame, timg);
        if (ret != SUCCESS){
            LOG_ERROR("yolov8 preprocess fail ...");
            return ret;
        }

        InferenceDataType infer_input_data;
        infer_input_data.emplace_back(
            std::make_pair((void *)timg.ptr<float>(), timg.total() * timg.elemSize())
        );
        engine_->BindingInput(infer_input_data);

        ret = postprocess(letterbox_frame, objects);
        if (ret != SUCCESS){
            LOG_ERROR("yolov8 postprocess fail ...");
            return ret;
        }   

        letterbox_decode(objects, letterbox_info_.hor, letterbox_info_.pad, task_type_);

        return SUCCESS;
    }

    std::shared_ptr<Model> CreateInferModel(const std::string &model_path, const TaskType task_type, bool use_plugin) {
        std::shared_ptr<ModelImpl> Instance(new ModelImpl());
        if (Instance->Load(model_path, task_type, use_plugin) != SUCCESS) {
            Instance.reset();
        }
        return Instance;
    }
} // namespace YOLOv8
