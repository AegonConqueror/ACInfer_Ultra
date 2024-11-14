
#include "engine.h"
#include <onnxruntime_cxx_api.h>

class ONNXEngine : public ACEngine {
public:
    ONNXEngine(): m_env(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "ONNXEngine")),
        memoryInfo(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {};

    ~ONNXEngine() override { destory(); };

    error_e create(const std::string &file);

    virtual void        Print() override;
    virtual void        BindingInput(InferenceDataType& inputData) override;
    virtual void        GetInferOutput(InferenceDataType& outputData) override;

    virtual std::vector<int>                GetInputShape(int index) override;
    virtual std::vector<std::vector<int>>   GetOutputShapes() override;
    virtual std::string                     GetInputType(int index) override;
    virtual std::vector<std::string>        GetOutputTypes() override;

private:
    error_e destory();

private:
    std::vector<ONNXTensorElementDataType>  input_types;
    std::vector<ONNXTensorElementDataType>  output_types;

    Ort::Session*                           m_session = nullptr;
    Ort::Env        						m_env;
    Ort::AllocatorWithDefaultOptions 		m_ortAllocator;

    uint8_t 								m_numInputs;
    uint8_t 								m_numOutputs;
    std::vector<char*> 						m_inputNodeNames;
    std::vector<char*> 						m_outputNodeNames;
    std::vector<int64_t> 					m_inputTensorSizes;
    std::vector<int64_t> 					m_outputTensorSizes;
    std::vector<std::vector<int64_t>> 		m_inputShapes;
    std::vector<std::vector<int64_t>> 		m_outputShapes;

    Ort::SessionOptions 					sessionOptions;
    Ort::MemoryInfo                         memoryInfo;

    std::vector<Ort::Value> inputTensors;
	std::vector<Ort::Value> outputTensors;
};


inline std::string data_type_string(ONNXTensorElementDataType dt){
    switch(dt){
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return "Float";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return "Float16";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return "UInt8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return "Int8";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return "Int32";
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return "Int64";
        default: return "Unknow";
    }
}

error_e ONNXEngine::destory() {
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
    m_inputShapes.clear();
    m_outputShapes.clear();
    m_inputTensorSizes.clear();
    m_outputTensorSizes.clear();

    for(int i = 0; i < m_inputNodeNames.size(); i++){
        free(m_inputNodeNames[i]);
    }
    m_inputNodeNames.clear();

    for(int i = 0; i < m_outputNodeNames.size(); i++){
        free(m_outputNodeNames[i]);
    }
    m_outputNodeNames.clear();
    return SUCCESS;
}

void ONNXEngine::Print() {
    LOG_INFO("****************************************************************************");
    LOG_INFO("Infer %p detail", this);
    LOG_INFO("\tInputs: %d", m_numInputs);
    for(int i = 0; i < m_numInputs; ++i){
        auto input_shape = m_inputShapes[i];
        LOG_INFO("\t\t%d.%s : shape {%s}, %s", i, m_inputNodeNames[i], iTools::vector_shape_string(input_shape).c_str(), data_type_string(input_types[i]).c_str());
    }

    LOG_INFO("\tOutputs: %d", m_numOutputs);
    for(int i = 0; i < m_numOutputs; ++i){
        auto output_shape = m_outputShapes[i];
        LOG_INFO("\t\t%d.%s : shape {%s}, %s", i, m_outputNodeNames[i], iTools::vector_shape_string(output_shape).c_str(), data_type_string(output_types[i]).c_str());
    }
    LOG_INFO("****************************************************************************");
}

std::vector<int> ONNXEngine::GetInputShape(int index) {
    std::vector<int> output;
    output.reserve(m_inputShapes[index].size());
    for (int64_t val : m_inputShapes[index]) {
        output.push_back(static_cast<int>(val));
    }
    return output;
}

std::vector<std::vector<int>> ONNXEngine::GetOutputShapes() {
    std::vector<std::vector<int>> output;
    for (const auto& vec64 : m_outputShapes) {
        std::vector<int> vec;
        for (const auto& val64 : vec64) {
            vec.push_back(static_cast<int>(val64));
        }
        output.push_back(vec);
    }
    return output;
}

std::string ONNXEngine::GetInputType(int index){
    return data_type_string(input_types[index]);
}

std::vector<std::string> ONNXEngine::GetOutputTypes(){
    std::vector<std::string> output_chars;
    for (auto output_type : output_types){
        output_chars.push_back(data_type_string(output_type));
    }
    return output_chars;
}

error_e ONNXEngine::create(const std::string &file) {

    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    sessionOptions.SetLogSeverityLevel(4);
    sessionOptions.SetIntraOpNumThreads(0);
    OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);

    m_session = new Ort::Session(m_env, file.c_str(), sessionOptions);
    
    m_numInputs = m_session->GetInputCount();
    m_inputNodeNames.reserve(m_numInputs);
    m_inputTensorSizes.reserve(m_numInputs);

    m_numOutputs = m_session->GetOutputCount();
    m_outputNodeNames.reserve(m_numOutputs);
    m_outputTensorSizes.reserve(m_numOutputs);


    input_types.clear();
    input_types.reserve(m_numOutputs);
    for (size_t i = 0; i < m_numInputs; i++) {
        Ort::TypeInfo typeInfo = m_session->GetInputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        input_types.push_back(tensorInfo.GetElementType());
        auto inputDims = tensorInfo.GetShape();
        inputDims[0] = 1;
        m_inputShapes.emplace_back(inputDims);

        const auto& curInputShape = m_inputShapes[i];

        m_inputTensorSizes.emplace_back(
            std::accumulate(std::begin(curInputShape), std::end(curInputShape), 1, std::multiplies<int64_t>())
        );
        auto inputName = m_session->GetInputNameAllocated(i, m_ortAllocator);
        m_inputNodeNames.emplace_back(strdup(inputName.get()));
    }

    output_types.clear();
    output_types.reserve(m_numOutputs);
    for (size_t i = 0; i < m_numOutputs; i++) {
        Ort::TypeInfo typeInfo = m_session->GetOutputTypeInfo(i);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        output_types.push_back(tensorInfo.GetElementType());
        m_outputShapes.emplace_back(tensorInfo.GetShape());

        const auto& curOutputShape = m_outputShapes[i];

        m_outputTensorSizes.emplace_back(
            std::accumulate(std::begin(curOutputShape), std::end(curOutputShape), 1, std::multiplies<int64_t>())
        );
        auto outputName = m_session->GetOutputNameAllocated(i, m_ortAllocator);
        m_outputNodeNames.emplace_back(strdup(outputName.get()));
    }
    return SUCCESS;
}

void ONNXEngine::BindingInput(InferenceDataType& inputData) {
    if (m_numInputs != inputData.size()) {
        LOG_ERROR("inputs num not match! inputs.size()=%ld, input_num_=%d", inputData.size(), m_numInputs);
        exit(1);
    }
    inputTensors.clear();
    inputTensors.reserve(m_numInputs);

    for (size_t i = 0; i < m_numInputs; i++) {
        auto input_type = input_types[i];
        if (input_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) { // 半精度io
            inputTensors.emplace_back(
                std::move(
                    Ort::Value::CreateTensor<Ort::Float16_t>(
                        memoryInfo, 
                        (Ort::Float16_t *)inputData[i].first,
                        m_inputTensorSizes[i], 
                        m_inputShapes[i].data(), 
                        m_inputShapes[i].size()
                    )
                )
            );
        } else { // 单精度io
            inputTensors.emplace_back(
                std::move(
                    Ort::Value::CreateTensor<float>(
                        memoryInfo, 
                        (float *)inputData[i].first,
                        m_inputTensorSizes[i], 
                        m_inputShapes[i].data(), 
                        m_inputShapes[i].size()
                    )
                )
            );
        }
    }
}

void ONNXEngine::GetInferOutput(InferenceDataType& outputData) {
    outputTensors = m_session->Run(
        Ort::RunOptions{nullptr}, 
        m_inputNodeNames.data(), 
        inputTensors.data(),
        m_numInputs, 
        m_outputNodeNames.data(), 
        m_numOutputs
    );

    outputData.reserve(m_numOutputs);

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

std::shared_ptr<ACEngine> create_engine(const std::string &file_path, bool use_plugins) {
    std::shared_ptr<ONNXEngine> Instance(new ONNXEngine());
    if (Instance->create(file_path) != SUCCESS) {
        Instance.reset();
    }
    return Instance;
}