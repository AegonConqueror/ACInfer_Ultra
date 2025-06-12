
#include "yolov8.h"

#include "engine.h"
#include "process/preprocess.h"
#include "process/yolov8_postprocess.h"

void yolov8_pose_decode_gpu(
    float** preds, int* d_size, 
    int* num_dets, int* det_classes, float* det_scores, float* det_boxes, float* det_keypoints,
    int input_w, int input_h, int image_w, int image_h, int class_num,
    float conf_thres, float nms_thres, int keypoint_num
);

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
        LetterBoxInfo               letterbox_info_;

        ac_engine_attr              input_attr_;
        ac_engine_attrs             output_attrs_;

        int                         model_class_num_;
    };

    error_e ModelImpl::Load(const std::string &model_path, const TaskType task_type, bool use_plugin) {
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
        return SUCCESS;
    }

    error_e ModelImpl::preprocess(const cv::Mat &src_frame, cv::Mat &dst_frame, cv::Mat &timg) {
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

    error_e ModelImpl::postprocess(const cv::Mat &frame, std::vector<yolov8_result> &objects) {
        int input_w     = input_attr_.dims[3];
        int input_h     = input_attr_.dims[2];
        int image_w     = frame.cols;
        int image_h     = frame.rows;

        int keypoint_num = 17;

        InferenceData infer_output_data;
        engine_->GetInferOutput(infer_output_data);

        size_t output_num = infer_output_data.size();

        void *output_data[output_num];

        std::vector<float> detectiont_rects;
        std::vector<std::map<int, KeyPoint>> pose_keypoints;

        std::vector<std::string> pose_names = {"reg1", "cls1", "reg2", "cls2", "reg3", "cls3", "ps1", "ps2", "ps3"};
            
        if (pose_names.size() != output_num) return LOAD_MODEL_FAIL;

        for (size_t i = 0; i < output_num; i++) {
            output_data[i] = (void *)infer_output_data[engine_->GetOutputIndex(pose_names[i])].first;
        }

        LOG_INFO("using gpu kernel decode pose");
        int output_size[output_num];
        for (size_t i = 0; i < output_num; i++) {
            output_size[i] = infer_output_data[engine_->GetOutputIndex(pose_names[i])].second;
        }

        int keepTopK = 20;
        
        int num_dets[1];
        int det_classes[1 * keepTopK]; 
        float det_scores[1 * keepTopK]; 
        float det_boxes[1 * keepTopK * 4];
        float det_keypoints[1 * keepTopK * 3 * keypoint_num];

        yolov8_pose_decode_gpu(
            (float **)output_data, output_size, 
            num_dets, det_classes, det_scores, det_boxes, det_keypoints,
            input_w, input_h, image_w, image_h, model_class_num_,
            0.45, 0.45, keypoint_num
        );

        for (int i = 0; i < num_dets[0]; i++) {
            yolov8_result dr;

            int classId = det_classes[i];
            float conf = det_scores[i];
            
            int xmin = int(det_boxes[i * 4 + 0]);
            int ymin = int(det_boxes[i * 4 + 1]);
            int xmax = int(det_boxes[i * 4 + 2]);
            int ymax = int(det_boxes[i * 4 + 3]);

            std::map<int, KeyPoint> kp_map;
            for (int k = 0; k < keypoint_num; k++) {
                KeyPoint kp;
                kp.x = det_keypoints[k * keypoint_num * 3 + i * 3 + 0];
                kp.y = det_keypoints[k * keypoint_num * 3 + i * 3 + 1];
                kp.score = det_keypoints[k * keypoint_num * 3 + i * 3 + 2];
                kp.id = k;
                kp_map[k] = kp;
            }

            dr.class_id = classId;
            dr.confidence = conf;
            dr.box = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
            dr.keypoints = kp_map;

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

        InferenceData infer_input_data;
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
