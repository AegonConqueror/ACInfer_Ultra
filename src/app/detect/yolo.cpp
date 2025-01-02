
#include "yolo.h"
#include "yolo_postprocess/postprocess.h"

namespace YOLO {

    class YoloDetector : public Detector {
    public:
        error_e load(const std::string &model_path, Type yolo_type, bool use_plugin);

        virtual error_e Run(const cv::Mat &frame, std::vector<detect_result> &objects) override;

    private:
        error_e preprocess(const cv::Mat &frame, cv::Mat &timg);
        error_e inference(InferenceDataType &input_data);
        error_e postprocess(const cv::Mat &frame, std::vector<detect_result> &objects);

    public:
        std::shared_ptr<ACEngine>   engine_;
        Type                        yolo_type_;
        std::vector<int>            input_shape_;
        int                         data_type_;
    };
    

    error_e YoloDetector::load(const std::string &model_path, Type yolo_type, bool use_plugins) {
        yolo_type_ = yolo_type;
        engine_ = create_engine(model_path, use_plugins);
        if (!engine_){
            LOG_ERROR("Deserialize onnx failed.");
            return LOAD_MODEL_FAIL;
        }
        engine_->Print();
        input_shape_    = engine_->GetInputShape();
        data_type_      = (engine_->GetInputType() == "Float") ? CV_32F : CV_16F;

        return SUCCESS;
    }

    error_e YoloDetector::preprocess(const cv::Mat &frame, cv::Mat &timg) {
        int input_h     = input_shape_[2];
        int input_w     = input_shape_[3];

        if (yolo_type_ == Type::V8) {
            cv::Mat src, Rgb;
            cv::cvtColor(frame, Rgb, cv::COLOR_BGR2RGB);
            cv::resize(Rgb, src, cv::Size(input_w, input_h));
            cv::dnn::blobFromImage(src, timg, 1.0 / 255, cv::Size(input_w, input_h));
            timg.convertTo(timg, data_type_);
        } else {
            LOG_ERROR("This YOLO version does not support yet!");
            return PROCESS_FAIL;
        }
        return SUCCESS;
    }

    error_e YoloDetector::inference(InferenceDataType &input_data) {
        engine_->BindingInput(input_data);
        return SUCCESS;
    }

    error_e YoloDetector::postprocess(const cv::Mat &frame, std::vector<detect_result> &objects) {
        if (yolo_type_ == Type::V8) {
            auto image_w    = frame.cols;
            auto image_h    = frame.rows;
            auto input_w    = input_shape_[2];
            auto input_h    = input_shape_[3];

            auto output_shapes = engine_->GetOutputShapes();
            auto class_num = output_shapes[1][1];

            InferenceDataType infer_result_data;
            engine_->GetInferOutput(infer_result_data);

            size_t output_num = infer_result_data.size();
            output_num = 6;
            

            if (output_num == 3) {
                class_num = output_shapes[2][2];
                float scale_x = image_w / input_w;
                float scale_y = image_h / input_h;
                
                int64_t* outputs_nms = (int64_t*)infer_result_data[0].first;
                int outputs_nms_size = infer_result_data[0].second;
                int obj_num = outputs_nms_size / (sizeof(int64_t) * 3);
                

                auto boxes_shape = output_shapes[1];
                uint16_t* boxes_f16 = (uint16_t* )infer_result_data[1].first;
                float* boxes = iTools::halfToFloat((void *)boxes_f16, boxes_shape);

                auto label_conf_shape = output_shapes[2];
                uint16_t* label_conf_f16 = (uint16_t* )infer_result_data[2].first;
                float* label_conf = iTools::halfToFloat((void *)label_conf_f16, label_conf_shape);

                for (size_t i = 0; i < obj_num; i++) {
                    int ibatch = outputs_nms[i*3];
                    int iclass = outputs_nms[i*3 + 1];
                    int ibox = outputs_nms[i*3 + 2];
                    float* bbox = &boxes[ibox * 4];
                    float conf = label_conf[ibox * class_num + iclass];

                    int xmin = static_cast<int>(bbox[0] * scale_x);
                    int ymin = static_cast<int>(bbox[1] * scale_y);
                    int xmax = static_cast<int>(bbox[2] * scale_x);
                    int ymax = static_cast<int>(bbox[3] * scale_y);

                    detect_result result; 
                    result.box = cv::Rect(
                        xmin, 
                        ymin, 
                        xmax - xmin, 
                        ymax - ymin
                    );
                    result.class_id = iclass;
                    result.confidence = conf;
                    objects.push_back(result);
                }

                delete[] boxes;
                delete[] label_conf;

            } else if (output_num == 6) {
                void* output_data[output_num];

                for (size_t i = 0; i < output_num; i++){
                    auto output_shape = output_shapes[i];
                    if (data_type_ == CV_16F) {
                        uint16_t* item = (uint16_t* )infer_result_data[i].first;
                        output_data[i] = (void*)iTools::halfToFloat((void *)item, output_shape);
                    } else if (data_type_ == CV_32F) {
                        output_data[i] = (void*)infer_result_data[i].first;
                    } else {
                        LOG_ERROR("This data type does not support yet!");
                        return PROCESS_FAIL;
                    }
                }

                std::vector<float> DetectiontRects;
                yolov8::PostprocessSpilt(
                    (float **)output_data, DetectiontRects,
                    input_w, input_h, class_num, 0.65
                );
                
                for (int i = 0; i < DetectiontRects.size(); i += 6){
                    int classId = int(DetectiontRects[i + 0]);
                    float conf = DetectiontRects[i + 1];
                    int xmin = int(DetectiontRects[i + 2] * float(image_w) + 0.5);
                    int ymin = int(DetectiontRects[i + 3] * float(image_h) + 0.5);
                    int xmax = int(DetectiontRects[i + 4] * float(image_w) + 0.5);
                    int ymax = int(DetectiontRects[i + 5] * float(image_h) + 0.5);

                    detect_result result; 
                    result.box = cv::Rect(xmin, ymin, xmax - xmin, ymax - ymin);
                    result.class_id = classId;
                    result.confidence = conf;
                    objects.push_back(result);
                }

                // 释放内存
                if (data_type_ == CV_16F) {
                    for (size_t i = 0; i < output_num; i++) {
                        delete[] output_data[i];
                        output_data[i] = nullptr;
                    }
                }
            }
        } else {
            LOG_ERROR("This YOLO version does not support yet!");
            return PROCESS_FAIL;
        }
        return SUCCESS;
    }

    error_e YoloDetector::Run(const cv::Mat &frame, std::vector<detect_result> &objects) {
        cv::Mat input_img;
        auto ret = preprocess(frame, input_img);
        if (ret != SUCCESS){
            LOG_ERROR("yolo preprocess fail ...");
            return ret;
        }
        
        InferenceDataType infer_input_data;
        if (data_type_ == CV_32F) {
            infer_input_data.emplace_back(
                std::make_pair((void *)input_img.ptr<float>(), input_img.total() * input_img.elemSize())
            );
        } else if (data_type_ == CV_16F){
            infer_input_data.emplace_back(
                std::make_pair((void *)input_img.ptr<uint16_t>(), input_img.total() * input_img.elemSize())
            );
        }else{
            LOG_ERROR("This data type does not support yet!");
            return PROCESS_FAIL;
        }

        ret = inference(infer_input_data);
        if (ret != SUCCESS){
            LOG_ERROR("yolov8s inference fail ...");
            return ret;
        }

        return postprocess(frame, objects);
    }

    std::shared_ptr<Detector> CreateDetector(const std::string &model_path, Type yolo_type, bool use_plugins) {
        std::shared_ptr<YoloDetector> Instance(new YoloDetector());
        if (Instance->load(model_path, yolo_type, use_plugins) != SUCCESS) {
            Instance.reset();
        }
        return Instance;
    }
} // namespace YOLO
