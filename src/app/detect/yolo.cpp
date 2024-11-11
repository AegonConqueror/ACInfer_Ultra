
#include "yolo.h"
#include "yolo_postprocess/postprocess.h"

namespace YOLO {
    
    Detector::Detector(const std::string &model_path, int platform, Type yolo_type, bool use_plugins)
        : use_plugins_(use_plugins), yolo_type_(yolo_type)
    {
        error_e ret = Initialize(&engine_, platform, model_path);
        input_shape_ = GetInputShape(engine_, 0);
    }

    Detector::~Detector() {
        error_e ret = Destory(engine_);
        engine_ = nullptr;
    }

    error_e Detector::Preprocess(const cv::Mat &frame, cv::Mat &timg) {
        int input_h     = input_shape_[2];
        int input_w     = input_shape_[3];

        if (yolo_type_ == Type::V8) {
            cv::Mat src, Rgb;
            cv::cvtColor(frame, Rgb, cv::COLOR_BGR2RGB);
            cv::resize(Rgb, src, cv::Size(input_w, input_h));
            cv::dnn::blobFromImage(src, timg, 1.0 / 255, cv::Size(input_w, input_h));
            int targetType = (GetInputType(engine_, 0) == "Float") ? CV_32F : CV_16F;
            timg.convertTo(timg, targetType);
        } else {
            LOG_ERROR("This YOLO version does not support yet!");
            return PROCESS_FAIL;
        }
        return SUCCESS;
    }   

    error_e Detector::Postprocess(const cv::Mat &frame, std::vector<detect_result> &objects) {
        if (yolo_type_ == Type::V8) {
            auto image_w    = frame.cols;
            auto image_h    = frame.rows;
            auto input_w    = input_shape_[2];
            auto input_h    = input_shape_[3];

            auto output_shapes = GetOutputShapes(engine_);

            InferenceDataType infer_result_data;
            GetInferOutput(engine_, infer_result_data);

            if (output_shapes.size() == 1) {
                /* code */
            } else if (output_shapes.size() == 6) {
                int class_num = output_shapes[1][1];

                size_t output_num = infer_result_data.size();
                std::vector<std::unique_ptr<float[]>> output_data;

                for (size_t i = 0; i < output_num; i++){
                    auto output_shape = output_shapes[i];
                    std::unique_ptr<float[]> pred;
                    if (GetInputType(engine_, 0) == "Float16") {
                        uint16_t* item = (uint16_t* )infer_result_data[i].first;
                        pred = std::unique_ptr<float[]>(iTools::halfToFloat((void *)item, output_shape));
                    } else {
                        pred = std::unique_ptr<float[]>((float *)infer_result_data[0].first);
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

    error_e Detector::Run(const cv::Mat &frame, std::vector<detect_result> &objects) {
        cv::Mat input_img;
        auto ret = Preprocess(frame, input_img);
        if (ret != SUCCESS){
            LOG_ERROR("yolo preprocess fail ...");
            return ret;
        }

        InferenceDataType infer_input_data;
        infer_input_data.emplace_back(
            std::make_pair((void *)input_img.ptr<uint16_t>(), input_img.total() * input_img.elemSize())
        );

        BindingInput(engine_, infer_input_data);

        return Postprocess(frame, objects);
    }

} // namespace YOLO
