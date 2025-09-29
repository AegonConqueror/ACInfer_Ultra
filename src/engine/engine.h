/**
 * *****************************************************************************
 * File name:   engine.h
 * 
 * @brief  common inference engine api
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

#include <types/error.h>
#include "tools/ac_utils.h"

typedef struct engine_attr{
    int64_t  n_dims;
    std::vector<int64_t> dims;
} ac_engine_attr;
using ac_engine_attrs = std::vector<ac_engine_attr>;

typedef int MemorySize;
using InferenceData = std::vector<std::pair<void *, MemorySize>>;

class ACEngine {
public:
    virtual ~ACEngine() {};

    virtual void Print() = 0;
    virtual void BindingInput(InferenceData& inputData) = 0;
    virtual void GetInferOutput(InferenceData& outputData, bool sync=true) = 0;

    virtual const ac_engine_attrs   GetInputAttrs()     = 0;
    virtual const ac_engine_attrs   GetOutputAttrs()    = 0;

    virtual int GetOutputIndex(const std::string name)  = 0;
};

std::shared_ptr<ACEngine> create_engine(const std::string& file_path, bool use_plugins=false);

#endif // ACINFER_ULTRA_ENGINE_H