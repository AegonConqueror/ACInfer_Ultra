
#include "engine.h"
#include <onnxruntime_cxx_api.h>

static ac_tensor_type_e engine_type_convert(ONNXTensorElementDataType type) {
    switch (type)
    {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return AC_TENSOR_FLOAT;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        return AC_TENSOR_FLOAT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        return AC_TENSOR_UINT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        return AC_TENSOR_INT8;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        return AC_TENSOR_INT16;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        return AC_TENSOR_INT32;
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return AC_TENSOR_INT64;
    default:
        LOG_ERROR("unsupported rknn type: %d\n", type);
        exit(1);
    }
}

static ONNXTensorElementDataType engine_type_convert(ac_tensor_type_e type) {
    switch (type)
    {
    case AC_TENSOR_FLOAT:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
    case AC_TENSOR_FLOAT16:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
    case AC_TENSOR_UINT8:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
    case AC_TENSOR_INT8:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
    case AC_TENSOR_INT16:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16;
    case AC_TENSOR_INT32:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
    case AC_TENSOR_INT64:
        return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
    default:
        LOG_ERROR("unsupported rknn type: %d\n", type);
        exit(1);
    }
}

static ac_engine_attr engine_tensor_attr_encode(
    const int index,
    const char *name,
    const Ort::TypeInfo &typeInfo
) {
    ac_engine_attr shape;

    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();

    auto type = tensorInfo.GetElementType();
    auto inputDims = tensorInfo.GetShape();
    inputDims[0] = 1;

    auto n_elems = std::accumulate(std::begin(inputDims), std::end(inputDims), 1, std::multiplies<int64_t>());

    shape.index = index;
    shape.name = name;
    shape.n_dims = inputDims.size();
    for (int i = 0; i < inputDims.size(); ++i) {
        shape.dims[i] = inputDims[i];
    }

    shape.n_elems = n_elems;
    shape.size = n_elems;
    
    shape.qnt_zp = 0;
    shape.qnt_scale = .0f;

    shape.type = engine_type_convert(type);
    shape.layout = AC_TENSOR_NCHW;

    return shape;
}

class ONNXEngine : public ACEngine {
public:
    ONNXEngine(): m_env(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNXEngine")),
        memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {};

    ~ONNXEngine() override { destory(); };

    error_e create(const std::string &file);

    virtual void        Print() override;
    virtual void        BindingInput(InferenceData &inputData) override;
    virtual void        GetInferOutput(InferenceData &outputData, bool sync) override;

    virtual const ac_engine_attrs   GetInputAttrs()     override;
    virtual const ac_engine_attrs   GetOutputAttrs()    override;

    virtual int GetOutputIndex(const std::string name) override;

private:
    void destory();

private:
    std::vector<ONNXTensorElementDataType>  input_types;
    std::vector<ONNXTensorElementDataType>  output_types;

    Ort::Session *m_session = nullptr;

    Ort::Env        						m_env;
    Ort::AllocatorWithDefaultOptions 		m_ortAllocator;

    std::vector<char *> 						inputNodeNames_;
    std::vector<char *> 						outputNodeNames_;

    Ort::SessionOptions 					sessionOptions;
    Ort::MemoryInfo                         memoryInfo;

    std::vector<Ort::Value> inputTensors;
	std::vector<Ort::Value> outputTensors;

    uint32_t input_num_;
    uint32_t output_num_;

    std::vector<ac_engine_attr> input_attrs_;
    std::vector<ac_engine_attr> output_attrs_;

    std::unordered_map<std::string, uint32_t> name_index_map_;
};

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

void ONNXEngine::destory() {
    for (int i = 0; i < inputTensors.size(); ++i) {
        inputTensors[i].release();
    }
    inputTensors.clear();

    for (int i = 0; i < outputTensors.size(); ++i) {
        outputTensors[i].release();
    }
    outputTensors.clear();

    memoryInfo.release();
    m_session->release();
    delete m_session;
    m_session = nullptr;
    sessionOptions.release();
    m_env.release();

    for(int i = 0; i < inputNodeNames_.size(); i++){
        free(inputNodeNames_[i]);
    }
    inputNodeNames_.clear();

    for(int i = 0; i < outputNodeNames_.size(); i++){
        free(outputNodeNames_[i]);
    }
    outputNodeNames_.clear();
}

void ONNXEngine::Print() {
    LOG_INFO("****************************************************************************");
    LOG_INFO("Infer %p detail", this);
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

error_e ONNXEngine::create(const std::string &file) {

    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    sessionOptions.SetLogSeverityLevel(4);
    sessionOptions.SetIntraOpNumThreads(0);
    OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);

    m_session = new Ort::Session(m_env, file.c_str(), sessionOptions);
    
    input_num_ = m_session->GetInputCount();
    inputNodeNames_.reserve(input_num_);
    input_attrs_.reserve(input_num_);

    // 输入属性
    for (size_t i = 0; i < input_num_; i++) {
        Ort::TypeInfo typeInfo = m_session->GetInputTypeInfo(i);
        auto inputName = m_session->GetInputNameAllocated(i, m_ortAllocator);
        inputNodeNames_.emplace_back(strdup(inputName.get()));
        input_attrs_.emplace_back(engine_tensor_attr_encode(i, strdup(inputName.get()), typeInfo));
    }

    output_num_ = m_session->GetOutputCount();
    outputNodeNames_.reserve(output_num_);
    output_attrs_.reserve(output_num_);

    // 输出属性
    for (size_t i = 0; i < output_num_; i++) {
        Ort::TypeInfo typeInfo = m_session->GetOutputTypeInfo(i);
        auto outputName = m_session->GetOutputNameAllocated(i, m_ortAllocator);
        outputNodeNames_.emplace_back(strdup(outputName.get()));
        output_attrs_.emplace_back(engine_tensor_attr_encode(i, strdup(outputName.get()), typeInfo));
    }

    for (size_t i = 0; i < output_num_; i++) {
        name_index_map_[output_attrs_[i].name] = i;
    }
    
    return SUCCESS;
}

void ONNXEngine::BindingInput(InferenceData& inputData) {
    if (input_num_ != inputData.size()) {
        LOG_ERROR("inputs num not match! inputs.size()=%ld, input_num_=%d", inputData.size(), input_num_);
        exit(1);
    }
    inputTensors.clear();
    inputTensors.reserve(input_num_);

    for (size_t i = 0; i < input_num_; i++) {
        auto input_type = engine_type_convert(input_attrs_[i].type);
        auto tensor_size = static_cast<int64_t>(input_attrs_[i].n_elems);
        std::vector<int64_t> input_shape(input_attrs_[i].dims, input_attrs_[i].dims + input_attrs_[i].n_dims);

        if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
            // 半精度io
            inputTensors.emplace_back(
                std::move(
                    Ort::Value::CreateTensor<Ort::Float16_t>(
                        memoryInfo, 
                        (Ort::Float16_t *)inputData[i].first,
                        tensor_size,
                        input_shape.data(), 
                        input_shape.size()
                    )
                )
            );
        } else {
            // 单精度io
            inputTensors.emplace_back(
                std::move(
                    Ort::Value::CreateTensor<float>(
                        memoryInfo, 
                        (float *)inputData[i].first,
                        tensor_size,
                        input_shape.data(), 
                        input_shape.size()
                    )
                )
            );
        }
    }
}

void ONNXEngine::GetInferOutput(InferenceData& outputData, bool sync) {
    outputTensors = m_session->Run(
        Ort::RunOptions{nullptr}, 
        inputNodeNames_.data(), 
        inputTensors.data(),
        input_num_,
        outputNodeNames_.data(), 
        output_num_
    );

    outputData.reserve(output_num_);

    for (auto& elem : outputTensors) {
        auto output_shape = elem.GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int> output;
        output.reserve(output_shape.size());
        for (int64_t val : output_shape) {
            output.push_back(static_cast<int>(val));
        }

        auto elem_size = iTools::vectorProduct(output);

        if (elem.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
            outputData.emplace_back(
                std::make_pair(elem.GetTensorMutableData<int64_t>(), elem_size * sizeof(int64_t))
            );
        }else if (elem.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16){
            outputData.emplace_back(
                std::make_pair(elem.GetTensorMutableData<uint16_t>(), elem_size * sizeof(uint16_t))
            );
        } else {
            outputData.emplace_back(
                std::make_pair(elem.GetTensorMutableData<float>(), elem_size * sizeof(float))
            );
        }
    }
}

const ac_engine_attrs ONNXEngine::GetInputAttrs() {
    return input_attrs_;
}

const ac_engine_attrs ONNXEngine::GetOutputAttrs() {
    return output_attrs_;
}

int ONNXEngine::GetOutputIndex(const std::string name) {
    auto it = name_index_map_.find(name);
    if (it != name_index_map_.end()) {
        return it->second;
    } else {
        return -1;
    }
}

std::shared_ptr<ACEngine> create_engine(const std::string &file_path, bool use_plugins) {
    std::shared_ptr<ONNXEngine> Instance(new ONNXEngine());
    if (Instance->create(file_path) != SUCCESS) {
        Instance.reset();
    }
    return Instance;
}