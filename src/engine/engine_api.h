/**
 * *****************************************************************************
 * File name:   engine_api.h
 * 
 * @brief  inference engine api
 * 
 * 
 * Created by Aegon on 2023-06-30
 * Copyright © 2023 House Targaryen. All rights reserved.
 * *****************************************************************************
 */

#ifndef ACINFER_ULTRA_ENGINE_API_H
#define ACINFER_ULTRA_ENGINE_API_H

#include <iostream>
#include <stdio.h>
#include <vector>

#include "engine.h"

#ifdef USE_ONNXRUNTIME
    #include "onnx_engine.h"
#endif

#ifdef USE_ATALS
    #include "atlas_engine.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

MS_DLL error_e Initialize(
    AC_HANDLE* handle, 
    int platform,
    const std::string &file_path, 
    bool owner_device,
    bool model_log = true,
    bool use_plugins = false
);

MS_DLL error_e Destory(AC_HANDLE handle);

MS_DLL void BindingInput(AC_HANDLE handle, InferenceDataType& inputData);

MS_DLL void GetInferOutput(AC_HANDLE handle, InferenceDataType& outputData);

MS_DLL std::vector<int> GetInputShape(AC_HANDLE handle, int index);

MS_DLL std::string GetInputType(AC_HANDLE handle, int index);

MS_DLL std::vector<std::vector<int>> GetOutputShapes(AC_HANDLE handle);

#ifdef __cplusplus
}
#endif

#endif // ACINFER_ULTRA_ENGINE_API_H