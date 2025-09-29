#include "trt_builder.h"

#include <fstream>
#include <iterator>

#include "NvInfer.h"
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include <NvInferRuntimePlugin.h>

#include <opencv2/opencv.hpp>

#include "trt/trt_cuda.h"
#include "trt/trt_tensor.h"

using namespace nvinfer1;

class Logger : public ILogger {
public:
	virtual void log(Severity severity, const char* msg) noexcept override {
		if (severity == Severity::kINTERNAL_ERROR) {
			LOG_ERROR("NVInfer INTERNAL_ERROR: %s", msg);
			abort();
		}else if (severity == Severity::kERROR) {
			LOG_ERROR("NVInfer: %s", msg);
		}
		else  if (severity == Severity::kWARNING) {
			LOG_WARNING("NVInfer: %s", msg);
		}
		else  if (severity == Severity::kINFO) {
			LOG_INFO("NVInfer: %s", msg);
		}
		else {
			LOG_DEBUG("%s", msg);
		}
	}
};
static Logger gLogger;

static std::vector<int> dims64_to_vec32(const Dims64& d64) {
    std::vector<int> v;
    v.reserve(d64.nbDims);
    for (int i = 0; i < d64.nbDims; ++i) {
        long long val = d64.d[i];
        if (val < 0) val = 1;
        if (val > INT_MAX) val = INT_MAX;
        v.push_back(static_cast<int>(val));
    }
    return v;
}

static std::string join_dims(const std::vector<int>& dims){
    std::stringstream output;
    char buf[64];
    const char* fmts[] = {"%d", " x %d"};
    for(int i = 0; i < dims.size(); ++i){
        snprintf(buf, sizeof(buf), fmts[i != 0], dims[i]);
        output << buf;
    }
    return output.str();
}

static std::string dims_str(const nvinfer1::Dims& dims){
    return join_dims(std::vector<int>(dims.d, dims.d + dims.nbDims));
}

static const char* activation_type_name(nvinfer1::ActivationType activation_type){
    switch(activation_type){
        case nvinfer1::ActivationType::kRELU: return "ReLU";
        case nvinfer1::ActivationType::kSIGMOID: return "Sigmoid";
        case nvinfer1::ActivationType::kTANH: return "TanH";
        case nvinfer1::ActivationType::kLEAKY_RELU: return "LeakyRelu";
        case nvinfer1::ActivationType::kELU: return "Elu";
        case nvinfer1::ActivationType::kSELU: return "Selu";
        case nvinfer1::ActivationType::kSOFTSIGN: return "Softsign";
        case nvinfer1::ActivationType::kSOFTPLUS: return "Parametric softplus";
        case nvinfer1::ActivationType::kCLIP: return "Clip";
        case nvinfer1::ActivationType::kHARD_SIGMOID: return "Hard sigmoid";
        case nvinfer1::ActivationType::kSCALED_TANH: return "Scaled tanh";
        case nvinfer1::ActivationType::kTHRESHOLDED_RELU: return "Thresholded ReLU";
    }
    return "Unknow activation type";
}

static const char* pooling_type_name(nvinfer1::PoolingType type){
    switch(type){
        case nvinfer1::PoolingType::kMAX:                   return "MaxPooling";
        case nvinfer1::PoolingType::kAVERAGE:               return "AveragePooling";
        case nvinfer1::PoolingType::kMAX_AVERAGE_BLEND:     return "MaxAverageBlendPooling";
    }
    return "Unknow pooling type";
}

static std::string layer_type_name(ILayer* layer){
    switch(layer->getType()){
        case LayerType::kCONVOLUTION:       return "Convolution";
        case LayerType::kCAST:              return "Cast";
        case LayerType::kLRN:               return "LRN";
        case LayerType::kSCALE:             return "Scale";
        case LayerType::kSOFTMAX:           return "SoftMax";
        case LayerType::kDECONVOLUTION:     return "Deconvolution";
        case LayerType::kCONCATENATION:     return "Concatenation";
        case LayerType::kELEMENTWISE:       return "Elementwise";
        case LayerType::kPLUGIN:            return "Plugin";
        case LayerType::kUNARY:             return "UnaryOp operation";
        case LayerType::kPADDING:           return "Padding";
        case LayerType::kSHUFFLE:           return "Shuffle";
        case LayerType::kREDUCE:            return "Reduce";
        case LayerType::kTOPK:              return "TopK";
        case LayerType::kGATHER:            return "Gather";
        case LayerType::kMATRIX_MULTIPLY:   return "Matrix multiply";
        case LayerType::kRAGGED_SOFTMAX:    return "Ragged softmax";
        case LayerType::kCONSTANT:          return "Constant";
        case LayerType::kIDENTITY:          return "Identity";
        case LayerType::kPLUGIN_V2:         return "PluginV2";
        case LayerType::kSLICE:             return "Slice";
        case LayerType::kSHAPE:             return "Shape";
        case LayerType::kPARAMETRIC_RELU:   return "Parametric ReLU";
        case LayerType::kRESIZE:            return "Resize";
        case LayerType::kACTIVATION: {
            IActivationLayer* act = (IActivationLayer*)layer;
            auto type = act->getActivationType();
            return activation_type_name(type);
        }
        case LayerType::kPOOLING: {
            IPoolingLayer* pool = (IPoolingLayer*)layer;
            return pooling_type_name(pool->getPoolingType());
        }
    }
    return "Unknow layer type";
}

static std::string layer_descript(nvinfer1::ILayer* layer){
    switch(layer->getType()){
        case nvinfer1::LayerType::kCONVOLUTION: {
            nvinfer1::IConvolutionLayer* conv = (nvinfer1::IConvolutionLayer*)layer;
            return iTools::str_format("channel: %d, kernel: %s, padding: %s, stride: %s, dilation: %s, group: %d", 
                conv->getNbOutputMaps(),
                dims_str(conv->getKernelSizeNd()).c_str(),
                dims_str(conv->getPaddingNd()).c_str(),
                dims_str(conv->getStrideNd()).c_str(),
                dims_str(conv->getDilationNd()).c_str(),
                conv->getNbGroups()
            );
        }
        case nvinfer1::LayerType::kPOOLING: {
            nvinfer1::IPoolingLayer* pool = (nvinfer1::IPoolingLayer*)layer;
            return iTools::str_format(
                "window: %s, padding: %s",
                dims_str(pool->getWindowSizeNd()).c_str(),
                dims_str(pool->getPaddingNd()).c_str()
            );   
        }
        case nvinfer1::LayerType::kDECONVOLUTION:{
            nvinfer1::IDeconvolutionLayer* conv = (nvinfer1::IDeconvolutionLayer*)layer;
            return iTools::str_format("channel: %d, kernel: %s, padding: %s, stride: %s, group: %d", 
                conv->getNbOutputMaps(),
                dims_str(conv->getKernelSizeNd()).c_str(),
                dims_str(conv->getPaddingNd()).c_str(),
                dims_str(conv->getStrideNd()).c_str(),
                conv->getNbGroups()
            );
        }
        case LayerType::kCAST:
        case nvinfer1::LayerType::kACTIVATION:
        case nvinfer1::LayerType::kPLUGIN:
        case nvinfer1::LayerType::kLRN:
        case nvinfer1::LayerType::kSCALE:
        case nvinfer1::LayerType::kSOFTMAX:
        case nvinfer1::LayerType::kCONCATENATION:
        case nvinfer1::LayerType::kELEMENTWISE:
        case nvinfer1::LayerType::kUNARY:
        case nvinfer1::LayerType::kPADDING:
        case nvinfer1::LayerType::kSHUFFLE:
        case nvinfer1::LayerType::kREDUCE:
        case nvinfer1::LayerType::kTOPK:
        case nvinfer1::LayerType::kGATHER:
        case nvinfer1::LayerType::kMATRIX_MULTIPLY:
        case nvinfer1::LayerType::kRAGGED_SOFTMAX:
        case nvinfer1::LayerType::kCONSTANT:
        case nvinfer1::LayerType::kIDENTITY:
        case nvinfer1::LayerType::kPLUGIN_V2:
        case nvinfer1::LayerType::kSLICE:
        case nvinfer1::LayerType::kSHAPE:
        case nvinfer1::LayerType::kPARAMETRIC_RELU:
        case nvinfer1::LayerType::kRESIZE:
            return "";
    }
    return "Unknow layer type";
}

static bool layer_has_input_tensor(nvinfer1::ILayer* layer){
    int num_input = layer->getNbInputs();
    for(int i = 0; i < num_input; ++i){
        auto input = layer->getInput(i);
        if(input == nullptr)
            continue;

        if(input->isNetworkInput())
            return true;
    }
    return false;
}

static bool layer_has_output_tensor(nvinfer1::ILayer* layer){
    int num_output = layer->getNbOutputs();
    for(int i = 0; i < num_output; ++i){

        auto output = layer->getOutput(i);
        if(output == nullptr)
            continue;

        if(output->isNetworkOutput())
            return true;
    }
    return false;
}

namespace TRT {

    const char* mode_type_string(Mode type) {
		switch (type) {
		case Mode::FP32:
			return "FP32";
		case Mode::FP16:
			return "FP16";
		case Mode::INT8:
			return "INT8";
		default:
			return "UnknowTRTMode";
		}
	}

    template <typename Calibrator>
    class CalibrationDataReader : public Calibrator {
    public:
        CalibrationDataReader(
            const std::vector<std::string>& imgfiles, const Dims dims, const std::string cache_file, const float* mean, const float* std
        ) : mImageFiles_(imgfiles), mInputDims_(dims), mCacheFileName_(cache_file) {
            memcpy(mMean_, mean, sizeof(mMean_));
		    memcpy(mStd_,  std,  sizeof(mStd_));
            mBatchSize_ = mInputDims_.d[0];
            mFromCache_ = false;
            mFiles_.resize(mBatchSize_);
            checkCudaRuntime(cudaStreamCreate(&mStream_));
        }

        virtual ~CalibrationDataReader(){ checkCudaRuntime(cudaStreamDestroy(mStream_)); }

        int32_t getBatchSize() const noexcept override { return mBatchSize_; }

        bool getBatch(void *bindings[], const char *names[], int nbBindings) noexcept override {
            if (mCurBatch_ + mBatchSize_ > mImageFiles_.size()) return false;

            for(int i = 0; i < mBatchSize_; ++i) {
                mFiles_[i] = mImageFiles_[mCurBatch_++];
            }
			
            if (!tensor_){
                auto dims32 = dims64_to_vec32(mInputDims_);
				tensor_.reset(new Tensor(dims32));
				tensor_->set_stream(mStream_);
				tensor_->set_workspace(std::make_shared<Memory>());
			}

            for(int i = 0; i < mFiles_.size(); ++i){

                auto image = cv::imread(mFiles_[i]);

                int width   = tensor_->width();
                int height  = tensor_->height();

                cv::resize(image, image, cv::Size(width, height));
                tensor_->to_cpu(false);
                float scale = 1 / 255.0;
                cv::Mat inputframe = image;
                if(CV_MAT_DEPTH(inputframe.type()) != CV_32F)
                    inputframe.convertTo(inputframe, CV_32F, scale);
                
                cv::Mat ms[3];
                for (int c = 0; c < 3; ++c)
                    ms[c] = cv::Mat(height, width, CV_32F, tensor_->cpu<float>(i, c));
                
                split(inputframe, ms);
                Assert((void*)ms[0].data == (void*)tensor_->cpu<float>(i));

                for (int c = 0; c < 3; ++c)
			        ms[c] = (ms[c] - mMean_[c]) / mStd_[c];
            }
            
            bindings[0] = tensor_->gpu();
            return true;
        }

        const void* readCalibrationCache(std::size_t &length) noexcept override {
            mCalibrationCache_.clear();
            std::ifstream input(mCacheFileName_, std::ios::binary);
            input >> std::noskipws;

            if (input.good()) {
                std::copy(
                    std::istream_iterator<char>(input), 
                    std::istream_iterator<char>(), 
                    std::back_inserter(mCalibrationCache_)
                );
            }

            length = mCalibrationCache_.size();

            return length ? mCalibrationCache_.data() : nullptr;
        }

        void writeCalibrationCache(const void *cache, std::size_t length) noexcept override {
            std::ofstream output(mCacheFileName_, std::ios::binary);
            output.write(reinterpret_cast<const char *>(cache), length);
        }

    private:
        float                       mMean_[3];
        float                       mStd_[3];

        bool                        mFromCache_{false};
        std::vector<std::string>    mFiles_;
        std::vector<std::string>    mImageFiles_;
        size_t                      mBatchSize_;
        Dims                        mInputDims_;
        
        std::shared_ptr<Tensor>     tensor_;
        std::vector<char>           mCalibrationCache_;
        std::string                 mCacheFileName_;

        int                         mCurBatch_{0};
        CUDAStream                  mStream_{nullptr};
    };

    typedef CalibrationDataReader<IInt8EntropyCalibrator2> EntropyCalibrator;
    typedef CalibrationDataReader<IInt8MinMaxCalibrator>   MinMaxCalibrator;

    class ACBuilderImpl : public ACBuilder {
    public:

        bool create(const std::string& onnx_path, const Mode mode, const QntConfig& qnt_cfg);

        virtual bool compile(const std::string& onnx_path) override;


    private:
        Mode        mode_;
        int         maxBatchSize_;
        CalibratorType  qnt_type_;

        float                       mMean_[3];
        float                       mStd_[3];

        std::shared_ptr<IBuilder>             builder_;

        std::string                 int8ImageDirectory_;
        std::string                 engineOuputPath_;
        std::string                 calibratorOuputPath_;

        std::vector<std::string>    calibratorImageFiles_;
    };

    bool ACBuilderImpl::create(const std::string& onnx_path, const Mode mode, const QntConfig& qnt_cfg) {
        mode_ = mode;
        maxBatchSize_ = qnt_cfg.max_batch;
        qnt_type_ = qnt_cfg.qnt_type;
        int8ImageDirectory_ = qnt_cfg.img_Dir;

        memcpy(mMean_, qnt_cfg.mean, sizeof(qnt_cfg.mean));
        memcpy(mStd_,  qnt_cfg.std,  sizeof(qnt_cfg.std));

        auto onnx_name = iFile::file_name(onnx_path, false);
        std::string engine_output_dir = "./engine/";
        std::string cache_output_dir = "./engine/calibration_cache/";
        iFile::mkdirs(cache_output_dir);

        engineOuputPath_ = iTools::str_format("%s/%s.engine", engine_output_dir.c_str(), onnx_name.c_str());
        calibratorOuputPath_ = iTools::str_format("%s/%s.cache", cache_output_dir.c_str(), onnx_name.c_str());

        if (mode == Mode::INT8) {
            calibratorImageFiles_ = iFile::find_files(qnt_cfg.img_Dir);
            if (calibratorImageFiles_.empty()){
                LOG_ERROR("Can not find any images(jpg/png/bmp/jpeg/tiff) from directory: %s", qnt_cfg.img_Dir.c_str());
                return false;
            }

            if (calibratorImageFiles_.size() < maxBatchSize_){
                LOG_WARNING(
                    "Too few images provided, %d[provided] < %d[max batch size], image copy will be performed",
                    calibratorImageFiles_.size(), maxBatchSize_
                );

                int old_size = calibratorImageFiles_.size();
                for(int i = old_size; i < maxBatchSize_; i++)
                    calibratorImageFiles_.push_back(calibratorImageFiles_[i % old_size]);
            }
        }

        LOG_INFO("Compile %s Onnx Model '%s'.", mode_type_string(mode), onnx_path.c_str());

        builder_ = make_nvshared(createInferBuilder(gLogger));
        if (!builder_){
            LOG_ERROR("Can not create builder.");
            return false;
        }

        return true;
    }
    
    bool ACBuilderImpl::compile(const std::string& onnx_path) {
        const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = make_nvshared(builder_->createNetworkV2(explicitBatch));
		auto onnxParser = make_nvshared(nvonnxparser::createParser(*network, gLogger));

        if (!onnxParser->parseFromFile(onnx_path.c_str(), 1)) {
            LOG_ERROR("Can not parse OnnX file: %s", onnx_path.c_str());
            return false;
        }

        auto inputTensor = network->getInput(0);
		auto inputDims = inputTensor->getDimensions();

        std::shared_ptr<EntropyCalibrator> int8EntropyCalibrator;
        std::shared_ptr<MinMaxCalibrator> int8MinMaxCalibrator;

        auto config = make_nvshared(builder_->createBuilderConfig());
        if (mode_ == Mode::FP16){
			if (!builder_->platformHasFastFp16())
				LOG_WARNING("Platform not have fast fp16 support");
			config->setFlag(BuilderFlag::kFP16);
		}else if (mode_ == Mode::INT8){
			if (!builder_->platformHasFastInt8())
				LOG_WARNING("Platform not have fast int8 support");
			config->setFlag(BuilderFlag::kINT8);
		}

        if (mode_ == Mode::INT8) {
            auto calibratorDims = inputDims;
            calibratorDims.d[0] = maxBatchSize_;
            LOG_INFO("Using image list[%d files]: %s", calibratorImageFiles_.size(), int8ImageDirectory_.c_str());
            if (qnt_type_ == CalibratorType::Entropy) {
                int8EntropyCalibrator.reset(new EntropyCalibrator(
                    calibratorImageFiles_, calibratorDims, calibratorOuputPath_, mMean_, mStd_
                ));
                config->setInt8Calibrator(int8EntropyCalibrator.get());
            } else {
                int8MinMaxCalibrator.reset(new MinMaxCalibrator(
                    calibratorImageFiles_, calibratorDims, calibratorOuputPath_, mMean_, mStd_
                ));
                config->setInt8Calibrator(int8MinMaxCalibrator.get());
            }
            
        }

        size_t maxWorkspaceSize = 1 << 30;

        LOG_INFO("Input shape is %s", join_dims(std::vector<int>(inputDims.d, inputDims.d + inputDims.nbDims)).c_str());
		LOG_INFO("Set max batch size = %d", maxBatchSize_);
		LOG_INFO("Set max workspace size = %.2f MB", maxWorkspaceSize / 1024.0f / 1024.0f);
		LOG_INFO("Base device: %s", iCUDA::device_description().c_str());

        int net_num_input = network->getNbInputs();
		LOG_INFO("Network has %d inputs:", net_num_input);
		std::vector<std::string> input_names(net_num_input);
		for(int i = 0; i < net_num_input; ++i){
			auto tensor = network->getInput(i);
			auto dims = tensor->getDimensions();
			auto dims_str = join_dims(std::vector<int>(dims.d, dims.d+dims.nbDims));
			LOG_INFO("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());

			input_names[i] = tensor->getName();
		}

		int net_num_output = network->getNbOutputs();
		LOG_INFO("Network has %d outputs:", net_num_output);
		for(int i = 0; i < net_num_output; ++i){
			auto tensor = network->getOutput(i);
			auto dims = tensor->getDimensions();
			auto dims_str = join_dims(std::vector<int>(dims.d, dims.d+dims.nbDims));
			LOG_INFO("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());
		}

		int net_num_layers = network->getNbLayers();
		LOG_INFO("Network has %d layers:", net_num_layers);
		for(int i = 0; i < net_num_layers; ++i){
			auto layer = network->getLayer(i);
			auto name = layer->getName();
			auto type_str = layer_type_name(layer);
			auto input0 = layer->getInput(0);
			if(input0 == nullptr) continue;
			
			auto output0 = layer->getOutput(0);
			auto input_dims = input0->getDimensions();
			auto output_dims = output0->getDimensions();
			bool has_input = layer_has_input_tensor(layer);
			bool has_output = layer_has_output_tensor(layer);
			auto descript = layer_descript(layer);
			type_str = iTools::align_blank(type_str, 18);
			auto input_dims_str = iTools::align_blank(dims_str(input_dims), 18);
			auto output_dims_str = iTools::align_blank(dims_str(output_dims), 18);
			auto number_str = iTools::align_blank(iTools::str_format("%d.", i), 4);

			const char* token = "      ";
			if(has_input)
				token = "  >>> ";
			else if(has_output)
				token = "  *** ";

			LOG_INFO("%s%s%s %s-> %s%s", token, 
				number_str.c_str(), 
				type_str.c_str(),
				input_dims_str.c_str(),
				output_dims_str.c_str(),
				descript.c_str()
			);
		}

        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, maxWorkspaceSize);

        auto profile = builder_->createOptimizationProfile();
        for(int i = 0; i < net_num_input; ++i){
			auto input = network->getInput(i);
			auto inDims = input->getDimensions();

            nvinfer1::Dims64 minD = inDims;
            nvinfer1::Dims64 optD = inDims;
            nvinfer1::Dims64 maxD = inDims;

            if (minD.nbDims > 0) minD.d[0] = 1;
            if (optD.nbDims > 0) optD.d[0] = (int64_t)maxBatchSize_;
            if (maxD.nbDims > 0) maxD.d[0] = (int64_t)maxBatchSize_;

			profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, minD);
			profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, optD);
			profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, maxD);
		}
        config->addOptimizationProfile(profile);

        LOG_INFO("Building engine...");
        auto time_start = iTime::timestamp_now();
        // 10.x：直接构建“序列化引擎”
        nvinfer1::IHostMemory* plan = builder_->buildSerializedNetwork(*network, *config);
        if (!plan) {
            LOG_ERROR("Failed to build serialized network.");
            return false;
        }

        LOG_INFO("Build done %lld ms !", iTime::timestamp_now() - time_start);
        // 写文件并释放 plan
        bool ok = iFile::save_file(engineOuputPath_, plan->data(), plan->size());
        delete plan;
        return ok;
    }

    std::shared_ptr<ACBuilder> create_builder(const std::string& onnx_path, const Mode mode, const QntConfig& qnt_cfg) {
        std::shared_ptr<ACBuilderImpl> Instance(new ACBuilderImpl());
        if (!Instance->create(onnx_path, mode, qnt_cfg)) {
            Instance.reset();
        }
        return Instance;
    }
} // namespace TRT