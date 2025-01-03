
#include "trt_cuda.h"

namespace iCUDA {

    bool check_runtime(cudaError_t e, const char* call, int line, const char *file){
        if (e != cudaSuccess) {
            LOG_ERROR("CUDA Runtime error %s # %s, code = %s [ %d ] in file %s:%d", call, cudaGetErrorString(e), cudaGetErrorName(e), e, file, line);
            return false;
        }
        return true;
    }

    bool check_device_id(int device_id){
        int device_count = -1;
        checkCudaRuntime(cudaGetDeviceCount(&device_count));
        if(device_id < 0 || device_id >= device_count){
            LOG_ERROR("Invalid device id: %d, count = %d", device_id, device_count);
            return false;
        }
        return true;
    }

    int current_device_id(){
        int device_id = 0;
        checkCudaRuntime(cudaGetDevice(&device_id));
        return device_id;
    }

    std::string device_capability(int device_id){
        cudaDeviceProp prop;
        checkCudaRuntime(cudaGetDeviceProperties(&prop, device_id));
        return iLog::format("%d.%d", prop.major, prop.minor);
    }

    std::string device_name(int device_id){
        cudaDeviceProp prop;
        checkCudaRuntime(cudaGetDeviceProperties(&prop, device_id));
        return prop.name;
    }

    std::string device_description(int device_id){
        cudaDeviceProp prop;
        size_t free_mem, total_mem;

        checkCudaRuntime(cudaGetDevice(&device_id));
        checkCudaRuntime(cudaGetDeviceProperties(&prop, device_id));
        checkCudaRuntime(cudaMemGetInfo(&free_mem, &total_mem));

        return iLog::format(
            "[ID %d]<%s>[arch %d.%d][GMEM %.2f GB/%.2f GB]",
            device_id, prop.name, prop.major, prop.minor, 
            free_mem / 1024.0f / 1024.0f / 1024.0f,
            total_mem / 1024.0f / 1024.0f / 1024.0f
        );
    }

    AutoDevice::AutoDevice(int device_id){
        cudaGetDevice(&old_);
        checkCudaRuntime(cudaSetDevice(device_id));
    }

    AutoDevice::~AutoDevice(){
        checkCudaRuntime(cudaSetDevice(old_));
    }
} // namespace iCUDA
