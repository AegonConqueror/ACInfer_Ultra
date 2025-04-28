
#include "trt_builder.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>
#include <NvInferRuntimePlugin.h>

#include "trt/trt_cuda.h"

class Logger : public nvinfer1::ILogger {
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

namespace TRT {
    
    Model::Model(const std::string& onnx_model_path) {
        this->onnx_model_path_ = onnx_model_path;
    }

    Model::Model(const char* onnx_model_path) {
        this->onnx_model_path_ = onnx_model_path;
    }

    std::string Model::onnxmodel() const { 
        return this->onnx_model_path_; 
    }

    std::string Model::descript() const{
        return iLog::format("Onnx Model '%s'", onnx_model_path_.c_str());
    }

    Model Model::onnx(const std::string& file){
        Model output;
        output.onnx_model_path_ = file;
        return output;
    }

    /* ---------------------------------------- 华丽的分割线 ----------------------------------------*/
    /**
	 * @brief 定义校准数据读取器 （模板类）
	 */
    template <typename Calibrator>
    class CalibrationDataReader : public Calibrator
    {
    public:
        CalibrationDataReader(
            const std::vector<std::string>& imagefiles, 
            nvinfer1::Dims dims, 
            const Int8Process& preprocess)
        {
            Assert(preprocess != nullptr);
            this->dims_ = dims;
            this->allimgs_ = imagefiles;
            this->preprocess_ = preprocess;
            this->fromCalibratorData_ = false;
            files_.resize(dims.d[0]);
            checkCudaRuntime(cudaStreamCreate(&stream_));
        }

        CalibrationDataReader(
            const std::vector<uint8_t>& calibratorDataCache, 
            nvinfer1::Dims dims, 
            const Int8Process& preprocess)
        {
            Assert(preprocess != nullptr);
            this->dims_ = dims;
            this->calibratorDataCache_ = calibratorDataCache;
            this->preprocess_ = preprocess;
            this->fromCalibratorData_ = false;
            files_.resize(dims.d[0]);
            checkCudaRuntime(cudaStreamCreate(&stream_));
        }

        virtual ~CalibrationDataReader(){
            checkCudaRuntime(cudaStreamDestroy(stream_));
        }

        int32_t getBatchSize() const noexcept override {
            return dims_.d[0];
        }

        bool next(){
            int batch_size = dims_.d[0];
                if (cursor_ + batch_size > allimgs_.size())
                    return false;

                for(int i = 0; i < batch_size; ++i)
                    files_[i] = allimgs_[cursor_++];

                if (!tensor_){
                    tensor_.reset(new Tensor(dims_.nbDims, dims_.d));
                    tensor_->set_stream(stream_);
                    tensor_->set_workspace(std::make_shared<TRTMemory>());
                }

                preprocess_(cursor_, allimgs_.size(), files_, tensor_);
                return true;
        }

        bool getBatch(void *bindings[], const char *names[], int nbBindings) noexcept override {
            if (!next()) return false;
            bindings[0] = tensor_->gpu();
            return true;
        }

        const std::vector<uint8_t>& getEntropyCalibratorData() {
                return calibratorDataCache_;
        }

        const void *readCalibrationCache(std::size_t &length) noexcept override{
            if (fromCalibratorData_) {
                    length = this->calibratorDataCache_.size();
                    return this->calibratorDataCache_.data();
                }
                length = 0;
                return nullptr;
        }

        void writeCalibrationCache(const void *cache, std::size_t length) noexcept override{
            calibratorDataCache_.assign((uint8_t*)cache, (uint8_t*)cache + length);
        }

    private:
            Int8Process preprocess_;
            std::vector<std::string> allimgs_;
            size_t batchCudaSize_ = 0;
            int cursor_ = 0;
            nvinfer1::Dims dims_;
            std::vector<std::string> files_;
            std::shared_ptr<Tensor> tensor_;
            std::vector<uint8_t> calibratorDataCache_;
            bool fromCalibratorData_ = false;
            CUDAStream stream_ = nullptr;
    };

    /* ---------------------------------------- 华丽的分割线 ----------------------------------------*/

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

    const char* mode_string(Mode type) {
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

    bool compile(
		Mode mode,
		unsigned int maxBatchSize,
		const Model& source,
		const std::string& saveto,
		Int8Process int8process,
		const std::string& int8ImageDirectory,
		const std::string& int8CalibratorCacheFile,
        CalibratorType calibratorType,
		const size_t maxWorkspaceSize
	){
        // int8量化必须要有数据预处理
        if (mode == Mode::INT8 && int8process == nullptr){
			LOG_ERROR("int8process must not nullptr, when in int8 mode.");
			return false;
		}

        bool                        hasCalibratorCache = false;
		std::vector<uint8_t>        calibratorCacheData;
		std::vector<std::string>    calibratorImageFiles;

        if (mode == Mode::INT8){
            if (!int8CalibratorCacheFile.empty()){
				if (iFile::exists(int8CalibratorCacheFile)){
					calibratorCacheData = iFile::load_file(int8CalibratorCacheFile);
					if (calibratorCacheData.empty()){
						LOG_ERROR("calibratorFile is set as: %s, but we read is empty.", int8CalibratorCacheFile.c_str());
						return false;
					}
					hasCalibratorCache = true;
				}
			}

            if (!hasCalibratorCache) {
                calibratorImageFiles = iFile::find_files(int8ImageDirectory);
				if (calibratorImageFiles.empty()){
					LOG_ERROR("Can not find any images(jpg/png/bmp/jpeg/tiff) from directory: %s", int8ImageDirectory.c_str());
						return false;
				}

				if (calibratorImageFiles.size() < maxBatchSize){

					LOG_WARNING("Too few images provided, %d[provided] < %d[max batch size], image copy will be performed", calibratorImageFiles.size(), maxBatchSize);

					int old_size = calibratorImageFiles.size();
					for(int i = old_size; i < maxBatchSize; i++)
						calibratorImageFiles.push_back(calibratorImageFiles[i % old_size]);
				}
            }
        }

        LOG_INFO("Compile %s %s.", mode_string(mode), source.descript().c_str());

        auto builder = make_nvshared(nvinfer1::createInferBuilder(gLogger));
		if (!builder){
            LOG_ERROR("Can not create builder.");
            return false;
        }

        auto config = make_nvshared(builder->createBuilderConfig());
        if (mode == Mode::FP16){
			if (!builder->platformHasFastFp16())
				LOG_WARNING("Platform not have fast fp16 support");
			config->setFlag(nvinfer1::BuilderFlag::kFP16);
		}else if (mode == Mode::INT8){
			if (!builder->platformHasFastInt8())
				LOG_WARNING("Platform not have fast int8 support");
			config->setFlag(nvinfer1::BuilderFlag::kINT8);
		}

        const auto explicitBatch = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
        auto network = make_nvshared(builder->createNetworkV2(explicitBatch));
		auto onnxParser = make_nvshared(nvonnxparser::createParser(*network, gLogger));

        if (!onnxParser->parseFromFile(source.onnxmodel().c_str(), 1)) {
            LOG_ERROR("Can not parse OnnX file: %s", source.onnxmodel().c_str());
            return false;
        }

        auto inputTensor = network->getInput(0);
		auto inputDims = inputTensor->getDimensions();

        using EntropyCalibrator = CalibrationDataReader<nvinfer1::IInt8EntropyCalibrator2>;
        using MinMaxCalibrator  = CalibrationDataReader<nvinfer1::IInt8MinMaxCalibrator>;

        std::shared_ptr<EntropyCalibrator> int8EntropyCalibrator;
        std::shared_ptr<MinMaxCalibrator> int8MinMaxCalibrator;

        if (mode == Mode::INT8) {
            auto calibratorDims = inputDims;
            calibratorDims.d[0] = maxBatchSize;
            if (hasCalibratorCache){
                LOG_INFO("Using exist entropy calibrator data[%d bytes]: %s", calibratorCacheData.size(), int8CalibratorCacheFile.c_str());
                if (calibratorType == CalibratorType::Entropy)
                    int8EntropyCalibrator.reset(
                        new EntropyCalibrator(calibratorCacheData, calibratorDims, int8process)
                    );
                else
                    int8MinMaxCalibrator.reset(
                        new MinMaxCalibrator(calibratorCacheData, calibratorDims, int8process)
                    );
            }else{
                LOG_INFO("Using image list[%d files]: %s", calibratorImageFiles.size(), int8ImageDirectory.c_str());
                if (calibratorType == CalibratorType::Entropy)
                    int8EntropyCalibrator.reset(
                        new EntropyCalibrator(calibratorImageFiles, calibratorDims, int8process)
                    );
                else
                    int8MinMaxCalibrator.reset(
                        new MinMaxCalibrator(calibratorImageFiles, calibratorDims, int8process)
                    );
            }
            if (calibratorType == CalibratorType::Entropy)
                config->setInt8Calibrator(int8EntropyCalibrator.get());
            else
                config->setInt8Calibrator(int8MinMaxCalibrator.get());
        }

        LOG_INFO("Input shape is %s", join_dims(std::vector<int>(inputDims.d, inputDims.d + inputDims.nbDims)).c_str());
		LOG_INFO("Set max batch size = %d", maxBatchSize);
		LOG_INFO("Set max workspace size = %.2f MB", maxWorkspaceSize / 1024.0f / 1024.0f);
		LOG_INFO("Base device: %s", iCUDA::device_description().c_str());

        // 显示输入信息
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

        // 显示输出信息
        int net_num_output = network->getNbOutputs();
        LOG_INFO("Network has %d outputs:", net_num_output);
        for(int i = 0; i < net_num_output; ++i){
			auto tensor = network->getOutput(i);
			auto dims = tensor->getDimensions();
			auto dims_str = join_dims(std::vector<int>(dims.d, dims.d+dims.nbDims));
			LOG_INFO("      %d.[%s] shape is %s", i, tensor->getName(), dims_str.c_str());
		}

        builder->setMaxBatchSize(maxBatchSize);
        config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, maxWorkspaceSize);

        auto profile = builder->createOptimizationProfile();
        for(int i = 0; i < net_num_input; ++i){
			auto input = network->getInput(i);
			auto input_dims = input->getDimensions();
			input_dims.d[0] = 1;
			profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, input_dims);
			profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, input_dims);
			input_dims.d[0] = maxBatchSize;
			profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, input_dims);
		}
        config->addOptimizationProfile(profile);

        LOG_INFO("Building engine...");
        auto time_start = iTime::timestamp_now();
        auto engine = make_nvshared(builder->buildEngineWithConfig(*network, *config));
        if (!engine) {
            LOG_INFO("Can not create engine.");
            return false;
        }

        if (mode == Mode::INT8 && !hasCalibratorCache) {
            if (!hasCalibratorCache){
                if (!int8CalibratorCacheFile.empty()){
                    LOG_INFO("Save calibrator to: %s", int8CalibratorCacheFile.c_str());

                    if (calibratorType == CalibratorType::Entropy)
                        iFile::save_file(int8CalibratorCacheFile, int8EntropyCalibrator->getEntropyCalibratorData());
                    else
                        iFile::save_file(int8CalibratorCacheFile, int8MinMaxCalibrator->getEntropyCalibratorData());
                    
                }else{
                    LOG_INFO("No set entropyCalibratorFile, and entropyCalibrator will not save.");
                }
            }
        }

        LOG_INFO("Build done %lld ms !", iTime::timestamp_now() - time_start);
        auto seridata = make_nvshared(engine->serialize());
        return iFile::save_file(saveto, seridata->data(), seridata->size());
    }

} // namespace TRT
