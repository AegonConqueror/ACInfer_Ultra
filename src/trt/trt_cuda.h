/**
 * *****************************************************************************
 * File name:   ac_cuda.h
 * 
 * @brief  Aegon cuda tools
 * 
 * 
 * Created by Aegon on 2023-04-18
 * Copyright © 2023 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACINFER_ULTRA_TRT_CUDA_H
#define ACINFER_ULTRA_TRT_CUDA_H

#include <cstring>
#include <string>
#include <memory>
#include <cuda.h>
#include <cuda_runtime.h>
#include "tools/ac_utils.h"

#define checkCudaRuntime(call) iCUDA::check_runtime(call, #call, __LINE__, __FILE__)

#define checkCudaKernel(...)                                                \
    __VA_ARGS__;                                                            \
    do{cudaError_t cudaStatus = cudaPeekAtLastError();                      \
        if (cudaStatus != cudaSuccess){                                     \
            LOG_ERROR("launch failed: %s", cudaGetErrorString(cudaStatus)); \
        }                                                                   \
    } while(0);

#define Assert(op)					                                        \
	do{                                                                     \
		bool cond = !(!(op));                                               \
		if(!cond){                                                          \
			LOG_ERROR("Assert failed, " #op);                               \
		}                                                                   \
	}while(false)

#define withDevice(device_id, func)                                         \
    do {                                                                    \
        int old_device;                                                     \
        cudaGetDevice(&old_device);                                         \
        cudaSetDevice(device_id);                                           \
        func();                                                             \
        cudaSetDevice(old_device);                                          \
    } while (0)

template<typename _T>
std::shared_ptr<_T> make_nvshared(_T* ptr){
    return std::shared_ptr<_T>(ptr, [](_T* p){ delete p;});
}

namespace iCUDA {

    bool check_runtime(cudaError_t e, const char* call, int iLine, const char* szFile);

    bool check_deviceId(int device_id);

    int current_deviceId();

    std::string device_description(int device_id = 0);

} // namespace iCUDA

#endif // ACINFER_ULTRA_TRT_CUDA_H