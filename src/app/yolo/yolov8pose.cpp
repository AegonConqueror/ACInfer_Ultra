
#include "yolov8pose.h"

#include "engine.h"

namespace POSEv8 {

    const float CONFIDENCE_THRESHOLD = 0.25;
    const float NMS_THRESHOLD = 0.45;
    const bool agnostic = false;

    struct ClassRes{
        int classes;
        float prob;
        std::vector< std::vector<float> > kpss;
    };

    struct BboxWithKeyPt : ClassRes{
        float x;
        float y;
        float w;
        float h;
    };

    struct KeyPtDetectRes {
        std::vector<BboxWithKeyPt> det_results;
    };

    void LetterBox(
        const cv::Mat& image, 
        cv::Mat& outImage,
        cv::Vec4d& params, //[ratio_x,ratio_y,dw,dh]
		const cv::Size& newShape = cv::Size(640, 640),
		bool autoShape = false,
		bool scaleFill = false,
		bool scaleUp = true,
		int stride = 32,
		const cv::Scalar& color = cv::Scalar(114, 114, 114)
    ) {
        cv::Size shape = image.size();
        float r = std::min((float)newShape.height / (float)shape.height, (float)newShape.width / (float)shape.width);
        if (!scaleUp) {
            r = std::min(r, 1.0f);
        }

        float ratio[2]{ r, r };
        int new_un_pad[2] = { (int)std::round((float)shape.width * r),(int)std::round((float)shape.height * r) };

        auto dw = (float)(newShape.width - new_un_pad[0]);
        auto dh = (float)(newShape.height - new_un_pad[1]);

        if (autoShape) {
            dw = (float)((int)dw % stride);
            dh = (float)((int)dh % stride);
        } else if (scaleFill) {
            dw = 0.0f;
            dh = 0.0f;
            new_un_pad[0] = newShape.width;
            new_un_pad[1] = newShape.height;
            ratio[0] = (float)newShape.width / (float)shape.width;
            ratio[1] = (float)newShape.height / (float)shape.height;
        }

        dw /= 2.0f;
        dh /= 2.0f;

        if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
            cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
        else
            outImage = image.clone();

        int top = int(std::round(dh - 0.1f));
        int bottom = int(std::round(dh + 0.1f));
        int left = int(std::round(dw - 0.1f));
        int right = int(std::round(dw + 0.1f));
        params[0] = ratio[0];
        params[1] = ratio[1];
        params[2] = left;
        params[3] = top;
        cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
    }

    float IOUCalculate(const BboxWithKeyPt &det_a, const BboxWithKeyPt &det_b) {
        cv::Point2f center_a(det_a.x, det_a.y);
        cv::Point2f center_b(det_b.x, det_b.y);
        cv::Point2f left_up(std::min(det_a.x - det_a.w / 2, det_b.x - det_b.w / 2),
                            std::min(det_a.y - det_a.h / 2, det_b.y - det_b.h / 2));
        cv::Point2f right_down(std::max(det_a.x + det_a.w / 2, det_b.x + det_b.w / 2),
                               std::max(det_a.y + det_a.h / 2, det_b.y + det_b.h / 2));
        float distance_d = (center_a - center_b).x * (center_a - center_b).x + (center_a - center_b).y * (center_a - center_b).y;
        float distance_c = (left_up - right_down).x * (left_up - right_down).x + (left_up - right_down).y * (left_up - right_down).y;
        float inter_l = det_a.x - det_a.w / 2 > det_b.x - det_b.w / 2 ? det_a.x - det_a.w / 2 : det_b.x - det_b.w / 2;
        float inter_t = det_a.y - det_a.h / 2 > det_b.y - det_b.h / 2 ? det_a.y - det_a.h / 2 : det_b.y - det_b.h / 2;
        float inter_r = det_a.x + det_a.w / 2 < det_b.x + det_b.w / 2 ? det_a.x + det_a.w / 2 : det_b.x + det_b.w / 2;
        float inter_b = det_a.y + det_a.h / 2 < det_b.y + det_b.h / 2 ? det_a.y + det_a.h / 2 : det_b.y + det_b.h / 2;
        if (inter_b < inter_t || inter_r < inter_l)
            return 0;
        float inter_area = (inter_b - inter_t) * (inter_r - inter_l);
        float union_area = det_a.w * det_a.h + det_b.w * det_b.h - inter_area;
        if (union_area == 0)
            return 0;
        else
            return inter_area / union_area - distance_d / distance_c;
    }

    void NmsDetect(std::vector<BboxWithKeyPt> &detections) {
        sort(detections.begin(), detections.end(), [=](const BboxWithKeyPt &left, const BboxWithKeyPt &right) {
            return left.prob > right.prob;
        });
    
        for (int i = 0; i < (int)detections.size(); i++)
            for (int j = i + 1; j < (int)detections.size(); j++)
            {
                if (detections[i].classes == detections[j].classes || agnostic) {
                    float iou = IOUCalculate(detections[i], detections[j]);
                    if (iou > NMS_THRESHOLD)
                        detections[j].prob = 0;
                }
            }
    
        detections.erase(std::remove_if(detections.begin(), detections.end(), [](const BboxWithKeyPt &det)
        { return det.prob == 0; }), detections.end());
    }

    class ModelImpl : public Model {
    public:
        error_e Load(const std::string &model_path, bool use_plugin);

        virtual error_e Run(const cv::Mat &frame, std::vector<yolov8_result> &objects) override;
    private:
        error_e preprocess(const cv::Mat &src_frame, cv::Mat &timg);
        error_e postprocess(std::vector<yolov8_result> &objects);

    private:
        std::shared_ptr<ACEngine>   engine_;
        std::vector<int>            input_shape_;
        cv::Vec4d                   letterbox_info_;

        ac_engine_attr              input_attr_;
        ac_engine_attrs             output_attrs_;

        int                         model_class_num_;
    };

    error_e ModelImpl::Load(const std::string &model_path, bool use_plugin) {
        engine_ = create_engine(model_path, use_plugin);
        if (!engine_){
            LOG_ERROR("Deserialize engine failed.");
            return LOAD_MODEL_FAIL;
        }

        engine_->Print();
        input_attr_    = engine_->GetInputAttrs()[0];
        output_attrs_ = engine_->GetOutputAttrs();

        model_class_num_ = output_attrs_[1].dims[1];
        return SUCCESS;
    }

    error_e ModelImpl::preprocess(const cv::Mat &src_frame, cv::Mat &timg) {
        int input_w     = input_shape_[3];
        int input_h     = input_shape_[2];
        
        float wh_ratio = (float)input_w / (float)input_h;

        cv::Mat dst_frame;
        LetterBox(src_frame, dst_frame, letterbox_info_, cv::Size(input_w, input_h));

        cv::Mat rgb_img, resize_img;
        cv::cvtColor(dst_frame, rgb_img, cv::COLOR_BGR2RGB);
        cv::resize(rgb_img, resize_img, cv::Size(input_w, input_h));
        cv::dnn::blobFromImage(resize_img, timg, 1.0 / 255, cv::Size(input_w, input_h));
        timg.convertTo(timg, CV_32F);
        
        return SUCCESS;
    }

    error_e ModelImpl::postprocess(std::vector<yolov8_result> &objects) {

        InferenceData infer_output_data;
        engine_->GetInferOutput(infer_output_data);

        auto rows = output_attrs_[0].dims[2];

        int nc = 2;
    	int nKeyCnt = 17;
        const int dimensions = 4 + nc + nKeyCnt * 3;

        cv::Mat output_pred_Mat = cv::Mat(dimensions, rows, CV_32FC1, infer_output_data[0].first);
        output_pred_Mat = output_pred_Mat.t();
        float* data = (float*)output_pred_Mat.data;

        KeyPtDetectRes detResult_oneImage;
    	cv::Vec4d params = letterbox_info_;
        
        for (int position = 0; position < rows; position++) {
            float* classes_scores = data + 4;
        	float* kps_ptr = data + 4 + nc;
            
            cv::Mat scores(1, nc, CV_32FC1, classes_scores);
			cv::Point class_id;
			double max_class_score=0;
			cv::minMaxLoc(scores, 0, &max_class_score, 0, &class_id);
			max_class_score = (float)max_class_score;

            max_class_score = (float)max_class_score;
			if (max_class_score > CONFIDENCE_THRESHOLD) {
                float x = (data[0] - params[2]) / params[0];
				float y = (data[1] - params[3]) / params[1];
				float w = data[2] / params[0];
				float h = data[3] / params[1];
				int left = std::max(int(x - 0.5 * w), 0);
				int top = std::max(int(y - 0.5 * h), 0);
				int width = int(w);
				int height = int(h);
				if (width <= 0 || height <= 0) {
					data += dimensions;
					continue;
				}

                BboxWithKeyPt box;
	           	box.prob = max_class_score;
	           	box.classes=class_id.x;
				box.x = x;// row[0] * ratio;
				box.y = y;//row[1] * ratio;
				box.w = w;//row[2] * ratio;
				box.h = h;// row[3] * ratio;

				for (int k=0; k< nKeyCnt; k++) {
					std::vector<float> kps;
					float kps_x = (*(kps_ptr + 3*k)   - params[2]) /params[0];
					float kps_y = (*(kps_ptr + 3*k + 1)  - params[3]) / params[1];
					float kps_s = *(kps_ptr + 3*k +2);
					kps.push_back(kps_x);
					kps.push_back(kps_y);
					kps.push_back(kps_s);
					box.kpss.push_back(kps);
				}

				detResult_oneImage.det_results.push_back(box);
            }
            data += dimensions;
        }
        NmsDetect(detResult_oneImage.det_results);
        for(auto res : detResult_oneImage.det_results){
            yolov8_result dr;
            dr.box = cv::Rect(res.x - res.w * 0.5, res.y - res.h * 0.5, res.w, res.h);
            dr.class_id = res.classes;
            dr.confidence = res.prob;

            std::map<int, KeyPoint> kps;
            for (int k = 0; k<res.kpss.size(); k++) {
                KeyPoint kp;
                auto ks = res.kpss[k];
                kp.x = ks[0];
                kp.y = ks[1];
                kp.score = ks[2];
                kp.id = k;
                kps.insert({k, kp});
            }
            dr.keypoints = kps;
            objects.push_back(dr);
        }
        return SUCCESS;
    }

    error_e ModelImpl::Run(const cv::Mat &frame, std::vector<yolov8_result> &objects) {
        cv::Mat letterbox_frame, timg;
        auto ret = preprocess(frame, timg);
        if (ret != SUCCESS){
            LOG_ERROR("yolov8 preprocess fail ...");
            return ret;
        }

        InferenceData infer_input_data;
        infer_input_data.emplace_back(
            std::make_pair((void *)timg.ptr<float>(), timg.total() * timg.elemSize())
        );
        engine_->BindingInput(infer_input_data);

        ret = postprocess(objects);
        if (ret != SUCCESS){
            LOG_ERROR("yolov8 postprocess fail ...");
            return ret;
        }   

        return SUCCESS;
    }

    std::shared_ptr<Model> CreateInferModel(const std::string &model_path, bool use_plugin) {
        std::shared_ptr<ModelImpl> Instance(new ModelImpl());
        if (Instance->Load(model_path, use_plugin) != SUCCESS) {
            Instance.reset();
        }
        return Instance;
    }


} // namespace POSEv8