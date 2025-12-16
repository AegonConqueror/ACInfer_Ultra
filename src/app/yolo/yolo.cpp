
#include "yolo.h"

#include "engine.h"

#include "process/warp_affine.h"
#include "kernel/preprocess_warp_affine.h"

namespace YOLO {
    class ModelImpl : public Model {
    public:
        ac_error_e Load(const std::string& model_path);

        virtual ac_error_e Run(const cv::Mat& frame, DETResults& results)  override;
    private:
        ac_error_e preprocess(const cv::Mat& src_img, float* timg);
        ac_error_e postprocess(DETResults& results);

    private:
        std::shared_ptr<ACEngine>   engine_;

        ac_engine_attr              input_attr_;
        ac_engine_attrs             output_attrs_;

        int                         input_width_;
        int                         input_height_;

        yolo::AffineMatrix          affine_;
    };

    ac_error_e ModelImpl::Load(const std::string& model_path) {
        engine_ = create_engine(model_path);
        if (!engine_){
            LOG_ERROR("Deserialize engine failed.");
            return AC_LOAD_MODEL_FAIL;
        }

        engine_->Print();
        input_attr_   = engine_->GetInputAttrs()[0];
        output_attrs_ = engine_->GetOutputAttrs();

        input_width_    = input_attr_.dims[3];
        input_height_   = input_attr_.dims[2];
        return AC_SUCCESS;
    }

    ac_error_e ModelImpl::preprocess(const cv::Mat& src_img, float* timg) {

        // AffineMatrix affine_matrix;
        affine_.compute(src_img.cols, src_img.rows, input_width_, input_height_);

        size_t src_size = src_img.cols * src_img.rows * 3;
        size_t dst_size = input_width_ * input_height_ * 3;

        uint8_t* psrc_device = nullptr;

        checkCudaRuntime(cudaMalloc(&psrc_device, src_size));
        checkCudaRuntime(cudaMemcpy(psrc_device, src_img.data, src_size, cudaMemcpyHostToDevice));

        yolo::AffineMatrix affine;
        affine.compute(src_img.cols, src_img.rows, input_width_, input_height_);

        float* d_affine_d2i = nullptr;
        checkCudaRuntime(cudaMalloc(&d_affine_d2i, sizeof(affine_.i2d)));
        checkCudaRuntime(cudaMemcpy(d_affine_d2i, affine_.d2i, sizeof(affine_.i2d), cudaMemcpyHostToDevice));

        ACKernel::Norm normalize = ACKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, ACKernel::ColorType::RGB);
        
        ACKernel::Norm* d_normalize = nullptr;
        checkCudaRuntime(cudaMalloc(&d_normalize, sizeof(normalize)));
        checkCudaRuntime(cudaMemcpy(d_normalize, &normalize, sizeof(normalize), cudaMemcpyHostToDevice));

        ACKernel::warp_affine_and_normalize_batchM_invoker(
            psrc_device, src_img.cols * 3, src_img.cols, src_img.rows,
            timg, input_width_ * 3, input_width_, input_height_,
            d_affine_d2i, d_normalize, 114
        );

        checkCudaRuntime(cudaPeekAtLastError());
       
        checkCudaRuntime(cudaFree(d_affine_d2i));
        checkCudaRuntime(cudaFree(psrc_device));

        return AC_SUCCESS;
    }

    ac_error_e ModelImpl::postprocess(DETResults& results) {
        InferenceData infer_output_data;
        engine_->GetInferOutput(infer_output_data);

        int* numDetectionsOutput    = (int *)infer_output_data[engine_->GetOutputIndex("NumDetections")].first;
        int* nmsClassesOutput       = (int *)infer_output_data[engine_->GetOutputIndex("DetectionClasses")].first;
        float* nmsScoresOutput      = (float *)infer_output_data[engine_->GetOutputIndex("DetectionScores")].first;
        float* nmsBoxesOutput       = (float *)infer_output_data[engine_->GetOutputIndex("DetectionBoxes")].first;

        for (int i = 0; i < numDetectionsOutput[0]; i++) {
            DETResult dr;

            int classId = nmsClassesOutput[i];
            float conf = nmsScoresOutput[i];

            int xmin = nmsBoxesOutput[i * 4 + 0];
            int ymin = nmsBoxesOutput[i * 4 + 1];
            int xmax = nmsBoxesOutput[i * 4 + 2];
            int ymax = nmsBoxesOutput[i * 4 + 3];

            xmin = int(xmin * affine_.d2i[0] + ymin * affine_.d2i[1] + affine_.d2i[2]);
            ymin = int(xmin * affine_.d2i[3] + ymin * affine_.d2i[4] + affine_.d2i[5]);
            xmax = int(xmax * affine_.d2i[0] + ymax * affine_.d2i[1] + affine_.d2i[2]);
            ymax = int(xmax * affine_.d2i[3] + ymax * affine_.d2i[4] + affine_.d2i[5]);

            dr.box = Box(xmin, ymin, xmax - xmin, ymax - ymin);
            dr.classId = classId;
            dr.score = conf;
            results.push_back(dr);
        }
        return AC_SUCCESS;
    }

    ac_error_e ModelImpl::Run(const cv::Mat& frame, DETResults& results) {
        float* timg = nullptr;
        checkCudaRuntime(cudaMalloc(&timg, input_width_ * input_height_ * 3 * sizeof(float)));
        auto ret = preprocess(frame, timg);
        if (ret != AC_SUCCESS) {
            LOG_ERROR(" yolo preprocess fail");
            return ret;
        }

        InferenceData infer_input_data;
        infer_input_data.emplace_back(
            std::make_pair((void *)timg, input_width_ * input_height_ * 3)
        );
        engine_->BindingInput(infer_input_data);

        ret = postprocess(results);
        if (ret != AC_SUCCESS) {
            LOG_ERROR(" yolo postprocess fail");
            return ret;
        }

        checkCudaRuntime(cudaFree(timg));
        return AC_SUCCESS;
    }

    std::shared_ptr<Model> CreateInference(const std::string& model_path) {
        std::shared_ptr<ModelImpl> Instance(new ModelImpl());
        if (Instance->Load(model_path) != AC_SUCCESS) {
            Instance.reset();
        }
        return Instance;
    }
    
} // namespace YOLO