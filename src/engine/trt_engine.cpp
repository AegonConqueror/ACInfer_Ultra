
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

typedef struct trt_attr{
    std::string name;
    std::string type;
    int64_t     index;
    int64_t     n_elems;
    std::vector<int64_t> dims;
} ac_trt_attr;

std::string data_type_string(nvinfer1::DataType dt){
    switch(dt){
        case nvinfer1::DataType::kFLOAT:    return "Float";
        case nvinfer1::DataType::kHALF:     return "Float16";
        case nvinfer1::DataType::kINT32:    return "UInt8";
        case nvinfer1::DataType::kUINT8:    return "Int8";
        default: return "Unknow";
    }
}

TRT::DataType convert_trt_datatype(nvinfer1::DataType dt){
    switch(dt){
        case nvinfer1::DataType::kFLOAT : return TRT::DataType::Float;
        case nvinfer1::DataType::kHALF  : return TRT::DataType::Float16;
        case nvinfer1::DataType::kINT32 : return TRT::DataType::Int32;
        case nvinfer1::DataType::kUINT8 : return TRT::DataType::UInt8;
        default:
            LOG_ERROR("Unsupport data type %d", dt);
            return TRT::DataType::Float;
    }
}

ac_trt_attr engine_tensor_attr_encode(
    int index,
    nvinfer1::Dims nvDims,
    nvinfer1::DataType nvType,
    const char* nodeName 
) {
    ac_trt_attr attr;

    attr.index = index;
    attr.name = nodeName;

    auto n_dims = nvDims.nbDims;
    int n_elems = 1;
    for (int i = 0; i < n_dims; ++i) {
        attr.dims.push_back(nvDims.d[i]);
        n_elems *= nvDims.d[i];
    }
    attr.n_elems = n_elems;
    attr.type = data_type_string(nvType);
    return attr;
}

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

    error_e create(const std::string &file);

    virtual void Print() override;
    virtual void BindingInput(InferenceData& inputData) override;
    virtual void GetInferOutput(InferenceData& outputData, bool sync) override;

    virtual const ac_engine_attrs GetInputAttrs()  override;
    virtual const ac_engine_attrs GetOutputAttrs() override;

    virtual int GetOutputIndex(const std::string name) override;

private:
    void destory();
    void synchronize();

    int get_max_batch_size();

private:
    int device_id_ = 0;

    std::vector<std::shared_ptr<TRT::Tensor>> inputs_;
    std::vector<std::shared_ptr<TRT::Tensor>> outputs_;
    std::vector<std::shared_ptr<TRT::Tensor>> orderdBlobs_;

    std::shared_ptr<EngineContext> context_;
    
    std::vector<void *> bindingsPtr_;

    uint32_t input_num_;
    uint32_t output_num_;

    std::vector<ac_trt_attr> input_attrs_;
    std::vector<ac_trt_attr> output_attrs_;

    std::unordered_map<std::string, uint32_t> name_index_map_;
};

void TRTEngine::destory() {
    int old_device = 0;
    checkCudaRuntime(cudaGetDevice(&old_device));
    checkCudaRuntime(cudaSetDevice(device_id_));
    this->context_.reset();
    this->inputs_.clear();
    this->outputs_.clear();
    this->input_attrs_.clear();
    this->output_attrs_.clear();
    checkCudaRuntime(cudaSetDevice(old_device));
}

void TRTEngine::Print() {
    LOG_INFO("****************************************************************************");
    LOG_INFO("Engine %p detail", this);
    LOG_INFO("\tBase device: %s", iCUDA::device_description().c_str());
    LOG_INFO("\tMax Batch Size: %d", this->get_max_batch_size());
    LOG_INFO("\tInputs: %d", input_num_);
    for (const auto& attr : input_attrs_) {
        LOG_INFO(
            "\t\t%d.%s : shape {%s}, %s, fmt NCHW, size %d", 
            attr.index, attr.name.c_str(), 
            iTools::vector_shape_string(attr.dims).c_str(), 
            attr.type.c_str(),
            attr.n_elems
        );
    }

    LOG_INFO("\tOutputs: %d", output_num_);
    for (const auto& attr : output_attrs_) {
        LOG_INFO(
            "\t\t%d.%s : shape {%s}, %s, fmt NCHW, size %d", 
            attr.index, attr.name.c_str(), 
            iTools::vector_shape_string(attr.dims).c_str(), 
            attr.type.c_str(),
            attr.n_elems
        );
    }
    LOG_INFO("****************************************************************************");
}

int TRTEngine::get_max_batch_size() {
    assert(this->context_ != nullptr);
    return this->context_->engine_->getMaxBatchSize();
}

void TRTEngine::synchronize() {
    checkCudaRuntime(cudaStreamSynchronize(context_->stream_));
}

error_e TRTEngine::create(const std::string &file) {
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
    outputs_.clear();

    input_attrs_.clear();
    output_attrs_.clear();

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
            input_attrs_.push_back(engine_tensor_attr_encode(i, dims, type, bindingName));
        }else{
            outputs_.push_back(newTensor);
            output_attrs_.push_back(engine_tensor_attr_encode(i, dims, type, bindingName));
        }
        orderdBlobs_.push_back(newTensor);
    }
    bindingsPtr_.resize(orderdBlobs_.size());
    input_num_ = input_attrs_.size();
    output_num_ = output_attrs_.size();

    for (size_t i = 0; i < output_num_; i++) {
        name_index_map_[output_attrs_[i].name] = i;
    }
    
    return SUCCESS;
}

const ac_engine_attrs TRTEngine::GetInputAttrs() {
    ac_engine_attrs attrs;
    for (const auto& attr : input_attrs_) {
        ac_engine_attr attr_;
        attr_.n_dims = attr.dims.size();
        attr_.dims = attr.dims;
        attrs.push_back(attr_);
    }
    
    return attrs;
}

const ac_engine_attrs TRTEngine::GetOutputAttrs() {
    ac_engine_attrs attrs;
    for (const auto& attr : output_attrs_) {
        ac_engine_attr attr_;
        attr_.n_dims = attr.dims.size();
        attr_.dims = attr.dims;
        attrs.push_back(attr_);
    }
    
    return attrs;
}

int TRTEngine::GetOutputIndex(const std::string name) {
    auto it = name_index_map_.find(name);
    if (it != name_index_map_.end()) {
        return it->second;
    } else {
        return -1;
    }
}

void TRTEngine::BindingInput(InferenceData& inputData) {
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

void TRTEngine::GetInferOutput(InferenceData& outputData, bool sync) {
    
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

    void **bindingsptr = bindingsPtr_.data();
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
        void *dataPtr = output->cpu();
        auto output_type = output->type();
        auto elem_size = iTools::vectorProduct(output->dims());
        if (output_type == TRT::DataType::Float) {
            outputData.push_back({dataPtr, elem_size});
        } else if (output_type == TRT::DataType::Float16) {
            outputData.push_back({dataPtr, elem_size});
        } else if (output_type == TRT::DataType::UInt8) {
            outputData.push_back({dataPtr, elem_size});
        } else if (output_type == TRT::DataType::Int32) {
            outputData.push_back({dataPtr, elem_size});
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
    if (Instance->create(file_path) != SUCCESS) {
        Instance.reset();
    }
    return Instance;
}