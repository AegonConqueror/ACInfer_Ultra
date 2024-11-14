
#include "yolo.h"
#include "yolo_postprocess/postprocess.h"

namespace YOLO {

    class YoloDetector : public Detector {
    public:
        error_e load(const std::string &model_path, Type yolo_type, bool use_plugin);

        virtual error_e Run(const cv::Mat &frame, std::vector<detect_result> &objects) override;

    private:
        error_e preprocess(const cv::Mat &frame);
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

    error_e YoloDetector::preprocess(const cv::Mat &frame) {
        int input_h     = input_shape_[2];
        int input_w     = input_shape_[3];

        cv::Mat timg;
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

        InferenceDataType infer_input_data;

        if (data_type_ == CV_32F) {
            infer_input_data.emplace_back(
                std::make_pair((void *)timg.ptr<float>(), timg.total() * timg.elemSize())
            );
        } else if (data_type_ == CV_16F){
            infer_input_data.emplace_back(
                std::make_pair((void *)timg.ptr<uint16_t>(), timg.total() * timg.elemSize())
            );
        }else{
            LOG_ERROR("This data type does not support yet!");
            return PROCESS_FAIL;
        }
        
        engine_->BindingInput(infer_input_data);
        return SUCCESS;
    }   

    error_e YoloDetector::postprocess(const cv::Mat &frame, std::vector<detect_result> &objects) {
        if (yolo_type_ == Type::V8) {
            auto image_w    = frame.cols;
            auto image_h    = frame.rows;
            auto input_w    = input_shape_[2];
            auto input_h    = input_shape_[3];

            auto output_shapes = engine_->GetOutputShapes();

            InferenceDataType infer_result_data;
            engine_->GetInferOutput(infer_result_data);

            if (output_shapes.size() == 1) {
                /* code */
            } else if (output_shapes.size() == 6) {
                int class_num = output_shapes[1][1];

                size_t output_num = infer_result_data.size();
                std::vector<std::unique_ptr<float[]>> output_data;

                for (size_t i = 0; i < output_num; i++){
                    auto output_shape = output_shapes[i];
                    std::unique_ptr<float[]> pred;
                    if (data_type_ == CV_16F) {
                        uint16_t* item = (uint16_t* )infer_result_data[i].first;
                        pred = std::unique_ptr<float[]>(iTools::halfToFloat((void *)item, output_shape));
                    } else if (data_type_ == CV_32F) {
                        pred = std::unique_ptr<float[]>((float *)infer_result_data[0].first);
                    } else {
                        LOG_ERROR("This data type does not support yet!");
                        return PROCESS_FAIL;
                    }
                    output_data.push_back(std::move(pred));
                }

                std::vector<float> DetectiontRects;
                yolov8::PostprocessSpilt(
                    (float **)output_data.data(), DetectiontRects,
                    input_w, input_h, class_num
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
            }
        } else {
            LOG_ERROR("This YOLO version does not support yet!");
            return PROCESS_FAIL;
        }
        return SUCCESS;
    }

    error_e YoloDetector::Run(const cv::Mat &frame, std::vector<detect_result> &objects) {
        auto ret = preprocess(frame);
        if (ret != SUCCESS){
            LOG_ERROR("yolo preprocess fail ...");
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
