
#include "engine.h"

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include "trt/trt_tensor.h"
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

class EngineContext {
public:
    ~EngineContext() { destroy(); }

    void set_stream(TRT::CUDAStream stream){
        if(owner_stream_){
            if (stream_) {cudaStreamDestroy(stream_);}
            owner_stream_ = false;
        }
        stream_ = stream;
    }

    bool build_context(const void* pdata, size_t size) {
        destroy();

        if(pdata == nullptr || size == 0)
            return false;

        owner_stream_ = true;
        checkCudaRuntime(cudaStreamCreate(&stream_));
        if(!stream_)
            return false;

        runtime_ = make_nvshared(nvinfer1::createInferRuntime(gLogger));
        if (!runtime_)
            return false;

        engine_ = make_nvshared(runtime_->deserializeCudaEngine(pdata, size, nullptr));
        if (!engine_)
            return false;

        context_ = make_nvshared(engine_->createExecutionContext());
        return context_ != nullptr;
    }

private:
    void destroy() {
        context_.reset();
        engine_.reset();
        runtime_.reset();

        if(owner_stream_){
            if (stream_) {cudaStreamDestroy(stream_);}
        }
        stream_ = nullptr;
    }

public:
    cudaStream_t stream_ = nullptr;
    bool owner_stream_ = false;
    std::shared_ptr<nvinfer1::IExecutionContext> context_;
    std::shared_ptr<nvinfer1::ICudaEngine> engine_;
    std::shared_ptr<nvinfer1::IRuntime> runtime_ = nullptr;
};

class TRTEngine : public ACEngine {
public:
    TRTEngine() {};

    ~TRTEngine() override { destory(); };

    error_e init(const std::string &file);

    virtual void        Print() override;
    virtual void        BindingInput(InferenceDataType& inputData) override;
    virtual void        GetInferOutput(InferenceDataType& outputData, bool sync) override;

    virtual std::vector<int>                GetInputShape(int index) override;
    virtual std::vector<std::vector<int>>   GetOutputShapes() override;
    virtual std::string                     GetInputType(int index) override;
    virtual std::vector<std::string>        GetOutputTypes() override;

private:
    void destory();
    void synchronize();
    int get_max_batch_size();

private:
    int device_id_ = 0;

    std::vector<std::shared_ptr<TRT::Tensor>> inputs_;
    std::vector<std::shared_ptr<TRT::Tensor>> outputs_;
    std::vector<std::string> inputs_name_;
    std::vector<std::string> outputs_name_;

    std::vector<std::shared_ptr<TRT::Tensor>> orderdBlobs_;

    std::shared_ptr<EngineContext> context_;
    std::vector<void* > bindingsPtr_;
};

static TRT::DataType convert_trt_datatype(nvinfer1::DataType dt){
    switch(dt){
        case nvinfer1::DataType::kFLOAT : return TRT::DataType::Float;
        case nvinfer1::DataType::kHALF  : return TRT::DataType::Float16;
        case nvinfer1::DataType::kINT32 : return TRT::DataType::Int32;
        default:
            LOG_ERROR("Unsupport data type %d", dt);
            return TRT::DataType::Float;
    }
}

void TRTEngine::destory() {
    int old_device = 0;
    checkCudaRuntime(cudaGetDevice(&old_device));
    checkCudaRuntime(cudaSetDevice(device_id_));
    this->context_.reset();
    this->inputs_.clear();
    this->outputs_.clear();
    this->inputs_name_.clear();
    this->outputs_name_.clear();
    checkCudaRuntime(cudaSetDevice(old_device));
}

int TRTEngine::get_max_batch_size() {
    assert(this->context_ != nullptr);
    return this->context_->engine_->getMaxBatchSize();
}

void TRTEngine::Print() {
    LOG_INFO("****************************************************************************");
    LOG_INFO("Infer %p detail", this);
    LOG_INFO("\tBase device: %s", iCUDA::device_description().c_str());
    LOG_INFO("\tMax Batch Size: %d", this->get_max_batch_size());
    LOG_INFO("\tInputs: %d", inputs_.size());
    for(int i = 0; i < inputs_.size(); ++i){
        auto& tensor = inputs_[i];
        auto& name = inputs_name_[i];
        LOG_INFO("\t\t%d.%s : shape {%s}, %s", i, name.c_str(), tensor->shape_string(), data_type_string(tensor->type()));
    }

    LOG_INFO("\tOutputs: %d", outputs_.size());
    for(int i = 0; i < outputs_.size(); ++i){
        auto& tensor = outputs_[i];
        auto& name = outputs_name_[i];
        LOG_INFO("\t\t%d.%s : shape {%s}, %s", i, name.c_str(), tensor->shape_string(), data_type_string(tensor->type()));
    }
    LOG_INFO("****************************************************************************");
}

void TRTEngine::synchronize() {
    checkCudaRuntime(cudaStreamSynchronize(context_->stream_));
}

error_e TRTEngine::init(const std::string &file) {
    auto engine_data = iFile::load_file(file);
    if (engine_data.empty()) {
        return FILE_READ_FAIL;
    }

    context_.reset(new EngineContext());
    if (!context_->build_context(engine_data.data(), engine_data.size())) {
        context_.reset();
        return LOAD_MODEL_FAIL;
    }

    cudaGetDevice(&device_id_);
    int nbBindings      = context_->engine_->getNbBindings();
    int max_batchsize   = context_->engine_->getMaxBatchSize();

    inputs_.clear();
    inputs_name_.clear();
    outputs_.clear();
    outputs_name_.clear();
    orderdBlobs_.clear();
    bindingsPtr_.clear();

    for (int i = 0; i < nbBindings; ++i) {
        auto dims                   = context_->engine_->getBindingDimensions(i);
        auto type                   = context_->engine_->getBindingDataType(i);
        const char* bindingName     = context_->engine_->getBindingName(i);

        dims.d[0] = max_batchsize;

        auto newTensor = std::make_shared<TRT::Tensor>(dims.nbDims, dims.d, convert_trt_datatype(type));
        newTensor->set_stream(context_->stream_);
        if (context_->engine_->bindingIsInput(i)){
            inputs_.push_back(newTensor);
            inputs_name_.push_back(bindingName);
        }else{
            outputs_.push_back(newTensor);
            outputs_name_.push_back(bindingName);
        }
        orderdBlobs_.push_back(newTensor);
    }
    bindingsPtr_.resize(orderdBlobs_.size());
    
    return SUCCESS;
}

std::vector<int> TRTEngine::GetInputShape(int index) {
    auto input = inputs_[index];
    return input->dims();
}

std::vector<std::vector<int>> TRTEngine::GetOutputShapes() {
    std::vector<std::vector<int>> outputshapes;
    for(auto output : outputs_) {
        outputshapes.push_back(output->dims());
    }
    return outputshapes;
}

std::string TRTEngine::GetInputType(int index){
    auto input = inputs_[index];
    return data_type_string(input->type());
}

std::vector<std::string> TRTEngine::GetOutputTypes(){
    std::vector<std::string> output_chars;
    for (auto output : outputs_){
        output_chars.push_back(data_type_string(output->type()));
    }
    return output_chars;
}


void TRTEngine::BindingInput(InferenceDataType& inputData) {
    if (inputs_.size() != inputData.size()) {
        LOG_ERROR("inputs num not match! inputs.size()=%ld, input_num_=%d", inputData.size(), inputs_.size());
        exit(1);
    }

    for (int i = 0; i < inputs_.size(); i++) {
        auto input_type = inputs_[i]->type();
        if (input_type == TRT::DataType::Float) {
            checkCudaRuntime(
                cudaMemcpy(inputs_[i]->gpu<float>(), inputData[i].first, inputData[i].second, cudaMemcpyHostToDevice)
            );
        } else if (input_type == TRT::DataType::Float16) {
            checkCudaRuntime(
                cudaMemcpy(inputs_[i]->gpu<uint16_t>(), inputData[i].first, inputData[i].second, cudaMemcpyHostToDevice)
            );
        } else if (input_type == TRT::DataType::UInt8) {
            checkCudaRuntime(
                cudaMemcpy(inputs_[i]->gpu<uint8_t>(), inputData[i].first, inputData[i].second, cudaMemcpyHostToDevice)
            );
        }
    }
}

void TRTEngine::GetInferOutput(InferenceDataType& outputData, bool sync) {
    
    int inputBatchSize = inputs_[0]->size(0);
    for(int i = 0; i < context_->engine_->getNbBindings(); ++i){
        auto dims = context_->engine_->getBindingDimensions(i);
        auto type = context_->engine_->getBindingDataType(i);
        dims.d[0] = inputBatchSize;
        if(context_->engine_->bindingIsInput(i)){
            context_->context_->setBindingDimensions(i, dims);
        }
    }

    for (int i = 0; i < outputs_.size(); ++i) {
        outputs_[i]->resize_single_dim(0, inputBatchSize);
        outputs_[i]->to_gpu(false);
    }

    for (int i = 0; i < orderdBlobs_.size(); ++i)
        bindingsPtr_[i] = orderdBlobs_[i]->gpu();

    void** bindingsptr = bindingsPtr_.data();
    bool execute_result = context_->context_->enqueueV2(bindingsptr, context_->stream_, nullptr);

    if(!execute_result){
        auto code = cudaGetLastError();
        LOG_ERROR("execute fail, code %d[%s], message %s", code, cudaGetErrorName(code), cudaGetErrorString(code));
        exit(1);
    }

    if (sync) {
        synchronize();
    }

    for (auto output : outputs_) {
        void* dataPtr = output->cpu();
        auto output_type = output->type();
        auto elem_size = iTools::vectorProduct(output->dims());
        if (output_type == TRT::DataType::Float) {
            outputData.push_back({dataPtr, elem_size * sizeof(float)});
        } else if (output_type == TRT::DataType::Float16) {
            outputData.push_back({dataPtr, elem_size * sizeof(uint16_t)});
        } else if (output_type == TRT::DataType::UInt8) {
            outputData.push_back({dataPtr, elem_size * sizeof(uint8_t)});
        } else if (output_type == TRT::DataType::Int32) {
            outputData.push_back({dataPtr, elem_size * sizeof(int32_t)});
        }
    }
}

std::shared_ptr<ACEngine> create_engine(const std::string &file_path, bool use_plugins) {
    if (use_plugins) {
        auto ret = initLibNvInferPlugins(&gLogger, "");
        if (!ret) {
			LOG_ERROR("init lib nvinfer plugins failed.");
		}
    }
    
    std::shared_ptr<TRTEngine> Instance(new TRTEngine());
    if (Instance->init(file_path) != SUCCESS) {
        Instance.reset();
    }
    return Instance;
}