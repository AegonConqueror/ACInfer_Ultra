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

#include "types/error.h"
#include "utils/utils.h"

typedef enum tensor_type : int {
    AC_TENSOR_FLOAT     = 0,
    AC_TENSOR_FLOAT16   = 1,
    AC_TENSOR_UINT8     = 2,
    AC_TENSOR_INT8      = 3,
    AC_TENSOR_INT16     = 4,
    AC_TENSOR_INT32     = 5,
    AC_TENSOR_INT64     = 6,
} ac_tensor_type_e;

typedef enum tensor_fmt {
    AC_TENSOR_NCHW      = 0,
    AC_TENSOR_NHWC      = 1,
    AC_TENSOR_OTHER     = 2,
    AC_TENSORT_UNKNOWN  = 3,
} ac_tensor_fmt_e;

typedef struct {
    std::string name;

    uint32_t index;
    uint32_t n_dims;
    uint32_t dims[4];
    uint32_t n_elems;
    uint32_t size;
    
    int     qnt_zp;
    float   qnt_scale;

    ac_tensor_type_e type;
    ac_tensor_fmt_e layout;
} ac_engine_attr;

typedef std::vector<ac_engine_attr> ac_engine_attrs;

typedef int MemorySize;
using InferenceData = std::vector<std::pair<void *, MemorySize>>;

class ACEngine {
public:
    virtual ~ACEngine() {};

    virtual void Print() = 0;
    virtual void BindingInput(InferenceData &inputData) = 0;
    virtual void GetInferOutput(InferenceData &outputData, bool sync=true) = 0;

    virtual const ac_engine_attrs   GetInputAttrs()     = 0;
    virtual const ac_engine_attrs   GetOutputAttrs()    = 0;

    virtual int GetOutputIndex(const std::string name)  = 0;
};

std::shared_ptr<ACEngine> create_engine(const std::string &file_path, bool use_plugins=false);

#endif // ACINFER_ULTRA_ENGINE_H