/**
 * *****************************************************************************
 * File name:   onnx_engine.h
 * 
 * @brief  ONNX Runtime inference engine
 * 
 * 
 * Created by Aegon on 2023-06-28
 * Copyright © 2023 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACINFER_ULTRA_ONNX_ENGINE_H
#define ACINFER_ULTRA_ONNX_ENGINE_H

#include "engine.h"

#include <onnxruntime_cxx_api.h>

class ONNXEngine : public ACEngine {
    public:
    ONNXEngine();
    ~ONNXEngine() override {};

    virtual error_e     Initialize(const std::string &file_path, bool use_plugins=false) override;
    virtual error_e     Destory() override;
    virtual void        BindingInput(InferenceDataType& inputData) override;
    virtual void        GetInferOutput(InferenceDataType& outputData) override;

    virtual void Print() override;

    virtual std::vector<int>                GetInputShape(int index) override;
    virtual std::vector<std::vector<int>>   GetOutputShapes() override;
    virtual std::string                     GetInputType(int index) override;
    virtual std::vector<std::string>        GetOutputTypes() override;

private:
    int                                     device_id_ = 0;
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

#endif // ACINFER_ULTRA_ONNX_ENGINE_H