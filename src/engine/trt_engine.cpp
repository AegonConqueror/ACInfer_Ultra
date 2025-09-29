#include "engine.h"

#include <algorithm>
#include <unordered_map>

#include <NvInfer.h>
#include <NvInferPlugin.h>
#include <NvOnnxParser.h>

#include "trt/trt_cuda.h"
#include "trt/trt_tensor.h"

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
			LOG_DEBUG("NVInfer: %s", msg);
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

/**
 * @brief  安全地把 Dims64 转成 vector<int>；动态维(<0)先按 1 兜底
 * 
 * @param   d64  [Dims64]
 */
static inline std::vector<int> dims64_to_vec32(const nvinfer1::Dims64& d64) {
    std::vector<int> v;
    v.reserve(d64.nbDims);
    for (int i = 0; i < d64.nbDims; ++i) {
        long long val = d64.d[i];
        if (val < 0) val = 1;                           // 动态维占位
        if (val > INT_MAX) val = INT_MAX;               // 防溢出
        v.push_back(static_cast<int>(val));
    }
    return v;
}

static inline std::string data_type_string(nvinfer1::DataType dt){
    switch(dt){
        case nvinfer1::DataType::kFLOAT:    return "Float";
        case nvinfer1::DataType::kHALF:     return "Float16";
        case nvinfer1::DataType::kINT32:    return "UInt8";
        case nvinfer1::DataType::kUINT8:    return "Int8";
        default: return "Unknow";
    }
}

static inline TRT::DataType convert_trt_datatype(nvinfer1::DataType dt){
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
    int index, nvinfer1::Dims nvDims, nvinfer1::DataType nvType, const char* nodeName 
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

class TRTEngine : public ACEngine {
public:

    ~TRTEngine() override { destory(); };

    ac_error_e create(const std::string &file);

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
    int             device_id_ = 0;

    cudaStream_t    stream_ = nullptr;
    bool            owner_stream_ = false;

    std::shared_ptr<nvinfer1::IExecutionContext>    context_;
    std::shared_ptr<nvinfer1::ICudaEngine>          engine_;
    std::shared_ptr<nvinfer1::IRuntime>             runtime_ = nullptr;

    std::vector<std::shared_ptr<TRT::Tensor>> inputs_;
    std::vector<std::shared_ptr<TRT::Tensor>> outputs_;
    std::vector<std::shared_ptr<TRT::Tensor>> orderdBlobs_;
    
    std::vector<void *> bindingsPtr_;

    uint32_t input_num_;
    uint32_t output_num_;

    std::vector<ac_trt_attr> input_attrs_;
    std::vector<ac_trt_attr> output_attrs_;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    std::unordered_map<std::string, uint32_t> name_index_map_;
};

void TRTEngine::destory() {
    int old_device = 0;
    checkCudaRuntime(cudaGetDevice(&old_device));
    checkCudaRuntime(cudaSetDevice(device_id_));
    context_.reset();
    engine_.reset();
    runtime_.reset();
    inputs_.clear();
    outputs_.clear();
    input_attrs_.clear();
    output_attrs_.clear();
    input_names_.clear();
    output_names_.clear();
    orderdBlobs_.clear();
    bindingsPtr_.clear();
    name_index_map_.clear();
    checkCudaRuntime(cudaSetDevice(old_device));

    if(owner_stream_)
        if (stream_) checkCudaRuntime(cudaStreamDestroy(stream_));
    stream_ = nullptr;
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

ac_error_e TRTEngine::create(const std::string &file) {
    auto engine_data = iFile::load_file(file);
    if (engine_data.empty()) {
        return AC_FILE_READ_FAIL;
    }

    owner_stream_ = true;
    checkCudaRuntime(cudaStreamCreate(&stream_));
    if(!stream_)
        return AC_LOAD_MODEL_FAIL;

    runtime_ = make_nvshared(nvinfer1::createInferRuntime(gLogger));
    if (!runtime_)
        return AC_LOAD_MODEL_FAIL;

    engine_ = make_nvshared(runtime_->deserializeCudaEngine(engine_data.data(), engine_data.size()));
    if (!engine_)
        return AC_LOAD_MODEL_FAIL;

    context_ = make_nvshared(engine_->createExecutionContext());
    if (!context_)
        return AC_LOAD_MODEL_FAIL;

    checkCudaRuntime(cudaGetDevice(&device_id_));

    inputs_.clear(); outputs_.clear();
    input_attrs_.clear(); output_attrs_.clear();
    orderdBlobs_.clear(); bindingsPtr_.clear();
    input_names_.clear(); output_names_.clear();
    name_index_map_.clear();

    int32_t nIO = engine_->getNbIOTensors();
    for (int32_t i = 0; i < nIO; ++i) {
        const char* tensorName = engine_->getIOTensorName(i);
        auto ioMode  = engine_->getTensorIOMode(tensorName);
        auto dtype   = engine_->getTensorDataType(tensorName);
        auto shape   = engine_->getTensorShape(tensorName); // 可能含 -1

        if (ioMode == nvinfer1::TensorIOMode::kINPUT) {
            auto optS = engine_->getProfileShape(tensorName, /*profileIndex=*/0, nvinfer1::OptProfileSelector::kOPT);
            if (optS.nbDims == shape.nbDims) {
                for (int d = 0; d < shape.nbDims; ++d)
                    if (shape.d[d] < 0) shape.d[d] = std::max<int64_t>(optS.d[d], 1);
            } else {
                // 兜底：把所有 -1 置为 1
                for (int d = 0; d < shape.nbDims; ++d)
                    if (shape.d[d] < 0) shape.d[d] = 1;
            }
        } else {
            // 输出：此时还没设输入形状，保持占位但把 -1 先置 1 以便分配最小缓存
            for (int d = 0; d < shape.nbDims; ++d)
                if (shape.d[d] < 0) shape.d[d] = 1;
        }

        auto dims32 = dims64_to_vec32(shape);
        auto newTensor = std::make_shared<TRT::Tensor>(
            dims32, convert_trt_datatype(dtype)
        );
        newTensor->set_stream(stream_);

        if (ioMode == nvinfer1::TensorIOMode::kINPUT) {
            inputs_.push_back(newTensor);
            input_attrs_.push_back(engine_tensor_attr_encode(i, shape, dtype, tensorName));
            input_names_.push_back(tensorName);
        } else {
            outputs_.push_back(newTensor);
            output_attrs_.push_back(engine_tensor_attr_encode(i, shape, dtype, tensorName));
            output_names_.push_back(tensorName);
            name_index_map_[tensorName] = (uint32_t)output_names_.size() - 1;
        }
        orderdBlobs_.push_back(newTensor);
    }

    bindingsPtr_.resize(orderdBlobs_.size(), nullptr);
    input_num_  = (uint32_t)input_attrs_.size();
    output_num_ = (uint32_t)output_attrs_.size();
    return AC_SUCCESS;
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
                cudaMemcpy(inputs_[i]->gpu<float>(), inputData[i].first, inputData[i].second, cudaMemcpyDeviceToDevice)
            );
        } else if (input_type == TRT::DataType::Float16) {
            checkCudaRuntime(
                cudaMemcpy(inputs_[i]->gpu<uint16_t>(), inputData[i].first, inputData[i].second, cudaMemcpyDeviceToDevice)
            );
        } else if (input_type == TRT::DataType::UInt8) {
            checkCudaRuntime(
                cudaMemcpy(inputs_[i]->gpu<uint8_t>(), inputData[i].first, inputData[i].second, cudaMemcpyDeviceToDevice)
            );
        }
    }
}

void TRTEngine::GetInferOutput(InferenceData& outputData, bool sync) {
    // 取真实 batch（来自第一个输入张量的当前大小）
    int inputBatchSize = inputs_.empty() ? 1 : (int)inputs_[0]->size(0);

    // 设置输入形状（仅当 batch 维或其他维度需要动态设置）
    for (size_t i = 0; i < input_names_.size(); ++i) {
        const char* name = input_names_[i].c_str();
        auto shape = engine_->getTensorShape(name);
        // 把 -1 替换为当前张量维度（从 TRT::Tensor 中取）
        auto& tin = inputs_[i];
        for (int d = 0; d < shape.nbDims; ++d) {
            int64_t cur = tin->size(d);
            if (d == 0) cur = inputBatchSize;
            if (shape.d[d] < 0) shape.d[d] = cur > 0 ? cur : 1;
        }
        context_->setInputShape(name, shape);
    }

    // 输出缓存尺寸与地址准备
    for (size_t i = 0; i < outputs_.size(); ++i) {
        outputs_[i]->resize_single_dim(0, inputBatchSize);
        outputs_[i]->to_gpu(false);
    }

    // 绑定所有 IO 张量的 device 指针
    // 注意：TRT10 使用 setTensorAddress / enqueueV3（替代 enqueueV2）【官方迁移指南】
    for (int32_t i = 0, e = engine_->getNbIOTensors(); i < e; ++i){
        const char* name = engine_->getIOTensorName(i);
        auto mode = engine_->getTensorIOMode(name);
        void* addr = nullptr;

        // 在 orderdBlobs_ 中的顺序与 IO 顺序一致（create() 已按 IO 顺序 push）
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            // 输入名在 input_names_ 的位置即对应 inputs_ 中张量
            auto it = std::find(input_names_.begin(), input_names_.end(), name);
            int idx = (int)std::distance(input_names_.begin(), it);
            addr = (void*)inputs_[idx]->gpu();
        } else {
            auto it = std::find(output_names_.begin(), output_names_.end(), name);
            int idx = (int)std::distance(output_names_.begin(), it);
            addr = (void*)outputs_[idx]->gpu();
        }
        context_->setTensorAddress(name, addr);
    }

    bool execute_result = context_->enqueueV3(stream_);
    if(!execute_result){
        auto code = cudaGetLastError();
        LOG_ERROR("execute fail, code %d[%s], message %s", code, cudaGetErrorName(code), cudaGetErrorString(code));
        exit(1);
    }

    if (sync) {
        synchronize();
    }

    // 回填输出 CPU 指针与元素数
    for (auto& output : outputs_) {
        void* dataPtr = output->cpu();
        auto output_type = output->type();
        auto elem_size = iTools::vector_shape_numel(output->dims());
        if (output_type == TRT::DataType::Float) {
            outputData.push_back({dataPtr, elem_size});
        } else if (output_type == TRT::DataType::Float16) {
            outputData.push_back({dataPtr, elem_size});
        } else if (output_type == TRT::DataType::UInt8) {
            outputData.push_back({dataPtr, elem_size});
        } else if (output_type == TRT::DataType::Int32) {
            outputData.push_back({dataPtr, elem_size});
        } else {
            LOG_ERROR("Unsupported output TRT::DataType");
            exit(1);
        }
    }
}

std::shared_ptr<ACEngine> create_engine(const std::string &file_path, bool use_plugins) {
    if (!iFile::exists(file_path)) {
        LOG_ERROR("Model file not exist: [%s]", file_path);
        exit(1);
    }

    if (use_plugins) {
        auto ret = initLibNvInferPlugins(&gLogger, "");
        if (!ret) {
			LOG_ERROR("init lib nvinfer plugins failed.");
		}
    }
    
    std::shared_ptr<TRTEngine> Instance(new TRTEngine());
    if (Instance->create(file_path) != AC_SUCCESS) {
        Instance.reset();
    }
    return Instance;
}
