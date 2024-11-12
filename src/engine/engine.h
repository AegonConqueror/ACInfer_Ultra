/**
 * *****************************************************************************
 * File name:   engine.h
 * 
 * @brief  common inference engine
 * 
 * 
 * Created by Aegon on 2023-06-28
 * Copyright © 2023 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACINFER_ULTRA_ENGINE_H
#define ACINFER_ULTRA_ENGINE_H

#include <string>
#include <memory>

#include "types/ACType.h"
#include "utils/utils.h"

typedef int MemorySize;
using InferenceDataType = std::vector<std::pair<void*, MemorySize>>;

class ACEngine {
public:
    virtual ~ACEngine() {};

    virtual error_e     Initialize(const std::string &file_path, bool owner_device, bool use_plugins=false) = 0;
    virtual error_e     Destory() = 0;
    virtual void        BindingInput(InferenceDataType& inputData) = 0;
    virtual void        GetInferOutput(InferenceDataType& outputData) = 0;

    virtual void Print() = 0;

    virtual std::vector<int>                GetInputShape(int index = 0) = 0;
    virtual std::vector<std::vector<int>>   GetOutputShapes() = 0;
    virtual std::string                     GetInputType(int index = 0) = 0;
    virtual std::vector<std::string>        GetOutputTypes() = 0;
};

std::shared_ptr<ACEngine> create_engine(const std::string &file, bool use_plugins=false);

#endif // ACINFER_ULTRA_ENGINE_H