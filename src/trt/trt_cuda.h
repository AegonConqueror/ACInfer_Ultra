
#ifndef ACINFER_ULTRA_TRT_CUDA_H
#define ACINFER_ULTRA_TRT_CUDA_H

#include <string>
#include <cuda.h>
#include <cuda_runtime.h>

#include "utils/utils.h"

#define checkCudaRuntime(call) iCUDA::check_runtime(call, #call, __LINE__, __FILE__)

#define checkCudaKernel(...)                                                                            \
    __VA_ARGS__;                                                                                        \
    do{cudaError_t cudaStatus = cudaPeekAtLastError();                                                  \
    if (cudaStatus != cudaSuccess){                                                                     \
        LOG_ERROR("launch failed: %s", cudaGetErrorString(cudaStatus));                                 \
    }} while(0);

#define Assert(op)					            \
	do{                                         \
		bool cond = !(!(op));                   \
		if(!cond){                              \
			LOG_ERROR("Assert failed, " #op);   \
		}                                       \
	}while(false)

template<typename _T>
std::shared_ptr<_T> make_nvshared(_T* ptr){
    return std::shared_ptr<_T>(ptr, [](_T* p){p->destroy();});
}

namespace iCUDA {

    bool check_runtime(cudaError_t e, const char* call, int iLine, const char *szFile);

    bool check_device_id(int device_id);

    int current_device_id();

    std::string device_capability(int device_id);

    std::string device_name(int device_id);

    std::string device_description(int device_id=0);

    class AutoDevice{
    public:
        AutoDevice(int device_id = 0);
        virtual ~AutoDevice();
    
    private:
        int old_ = -1;
    };

} // namespace iCUDA

#endif // ACINFER_ULTRA_TRT_CUDA_H