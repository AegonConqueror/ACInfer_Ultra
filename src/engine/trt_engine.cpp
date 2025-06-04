
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

static ac_tensor_type_e engine_type_convert(TRT::DataType type) {
    switch (type)
    {
    case TRT::DataType::Float:
        return AC_TENSOR_FLOAT;
    case TRT::DataType::Float16:
        return AC_TENSOR_FLOAT16;
    case TRT::DataType::UInt8:
        return AC_TENSOR_UINT8;
    case TRT::DataType::Int32:
        return AC_TENSOR_INT32;
    default:
        LOG_ERROR("unsupported rknn type: %d\n", type);
        exit(1);
    }
}

static ac_engine_attr engine_tensor_attr_encode(
    const int index,
    const char *tensor_name,
    const std::shared_ptr<TRT::Tensor> &trt_tensor) {
    ac_engine_attr shape;

    shape.index = index;
    shape.name = tensor_name;
    shape.n_dims = trt_tensor->ndims();
    for (int i = 0; i < trt_tensor->ndims(); ++i) {
        shape.dims[i] = trt_tensor->dims()[i];
    }
    shape.n_elems = trt_tensor->numel();
    shape.size = trt_tensor->numel();

    shape.qnt_zp = 0;
    shape.qnt_scale = .0f;

    shape.type = engine_type_convert(trt_tensor->type());
    shape.layout = AC_TENSOR_NCHW;

    return shape;
}

class TRTEngine : public ACEngine {
public:
    TRTEngine() {};

    ~TRTEngine() override { destory(); };

    error_e create(const std::string &file);

    virtual void        Print() override;
    virtual void        BindingInput(InferenceData& inputData) override;
    virtual void        GetInferOutput(InferenceData& outputData, bool sync) override;

    virtual const ac_engine_attrs   GetInputAttrs()     override;
    virtual const ac_engine_attrs   GetOutputAttrs()    override;

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

    std::vector<ac_engine_attr> input_attrs_;
    std::vector<ac_engine_attr> output_attrs_;
};

static TRT::DataType convert_trt_datatype(nvinfer1::DataType dt){
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

inline std::string data_type_string(ac_tensor_type_e dt){
    switch(dt){
        case AC_TENSOR_FLOAT:   return "Float";
        case AC_TENSOR_FLOAT16: return "Float16";
        case AC_TENSOR_UINT8:   return "UInt8";
        case AC_TENSOR_INT8:    return "Int8";
        case AC_TENSOR_INT16:   return "Int16";
        case AC_TENSOR_INT32:   return "Int32";
        case AC_TENSOR_INT64:   return "Int64";
        default: return "Unknow";
    }
}

inline std::string data_format_string(ac_tensor_fmt_e dt){
    switch(dt){
        case AC_TENSOR_NCHW:   return "NCHW";
        case AC_TENSOR_NHWC:   return "NHWC";
        default: return "Unknow";
    }
}

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

int TRTEngine::get_max_batch_size() {
    assert(this->context_ != nullptr);
    return this->context_->engine_->getMaxBatchSize();
}

void TRTEngine::Print() {
    LOG_INFO("****************************************************************************");
    LOG_INFO("Infer %p detail", this);
    LOG_INFO("\tBase device: %s", iCUDA::device_description().c_str());
    LOG_INFO("\tMax Batch Size: %d", this->get_max_batch_size());
    LOG_INFO("\tInputs: %d", input_num_);
    for(int i = 0; i < input_num_; ++i){
        auto input_attr = input_attrs_[i];
        std::vector<int64_t> shapes_(input_attr.dims, input_attr.dims + input_attr.n_dims);
        LOG_INFO(
            "\t\t%d.%s : shape {%s}, %s, fmt %s, size %d", 
            i, input_attrs_[i].name.c_str(), 
            iTools::vector_shape_string(shapes_).c_str(), 
            data_type_string(input_attr.type).c_str(),
            data_format_string(input_attr.layout).c_str(),
            input_attr.size
        );
    }

    LOG_INFO("\tOutputs: %d", output_num_);
    for(int i = 0; i < output_num_; ++i){
        auto output_attr = output_attrs_[i];
        std::vector<int64_t> shapes_(output_attr.dims, output_attr.dims + output_attr.n_dims);
        LOG_INFO(
            "\t\t%d.%s : shape {%s}, %s, fmt %s, size %d", 
            i, output_attrs_[i].name.c_str(), 
            iTools::vector_shape_string(shapes_).c_str(), 
            data_type_string(output_attr.type).c_str(),
            data_format_string(output_attr.layout).c_str(),
            output_attr.size
        );
    }
    LOG_INFO("****************************************************************************");
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
        const char *bindingName     = context_->engine_->getBindingName(i);

        dims.d[0] = max_batchsize;

        auto newTensor = std::make_shared<TRT::Tensor>(dims.nbDims, dims.d, convert_trt_datatype(type));
        newTensor->set_stream(context_->stream_);
        if (context_->engine_->bindingIsInput(i)){
            inputs_.push_back(newTensor);
            input_attrs_.push_back(engine_tensor_attr_encode(i, bindingName, newTensor));
        }else{
            outputs_.push_back(newTensor);
            output_attrs_.push_back(engine_tensor_attr_encode(i, bindingName, newTensor));
        }
        orderdBlobs_.push_back(newTensor);
    }
    bindingsPtr_.resize(orderdBlobs_.size());
    input_num_ = input_attrs_.size();
    output_num_ = output_attrs_.size();
    
    return SUCCESS;
}

const ac_engine_attrs TRTEngine::GetInputAttrs() {
    return input_attrs_;
}

const ac_engine_attrs TRTEngine::GetOutputAttrs() {
    return output_attrs_;
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
    if (Instance->create(file_path) != SUCCESS) {
        Instance.reset();
    }
    return Instance;
}