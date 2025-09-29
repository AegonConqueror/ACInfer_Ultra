#include "trt_memory.h"

#include "trt_cuda.h"

namespace TRT {
    
    static inline int get_device(int device_id) {
        if(device_id != CURRENT_DEVICE_ID){
            iCUDA::check_deviceId(device_id);
            return device_id;
        }

        checkCudaRuntime(cudaGetDevice(&device_id));
        return device_id;
    }

    CudaDeviceGuard::CudaDeviceGuard(int device_id) {
        checkCudaRuntime(cudaGetDevice(&old_device_));
        if (device_id != old_device_)
            checkCudaRuntime(cudaSetDevice(device_id));
    }

    CudaDeviceGuard::~CudaDeviceGuard() {
        checkCudaRuntime(cudaSetDevice(old_device_));
    }

    Memory::Memory(int device_id) {
        device_id_ = get_device(device_id);
    }

    Memory::Memory(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size) {
        reference_data(cpu, cpu_size, gpu, gpu_size);
    }

    Memory::~Memory() {
        release_all();
    }

    void* Memory::cpu(size_t size) {
        if (cpu_size_ < size) {
            release_cpu();
            cpu_size_ = size;
            CudaDeviceGuard guard(device_id_);
            checkCudaRuntime(cudaMallocHost(&cpu_, size));
            Assert(cpu_ != nullptr);
            memset(cpu_, 0, size);
        }
        return cpu_;
    }

    void* Memory::gpu(size_t size) {
        if (gpu_size_ < size) {
            release_gpu();
            gpu_size_ = size;
            CudaDeviceGuard guard(device_id_);
            checkCudaRuntime(cudaMalloc(&gpu_, size));
            checkCudaRuntime(cudaMemset(gpu_, 0, size));
        }
        return gpu_;
    }

    void Memory::release_gpu() {
        if (gpu_) {
            if (owner_gpu_) {
                CudaDeviceGuard guard(device_id_);
                checkCudaRuntime(cudaFree(gpu_));
            }
            gpu_ = nullptr;
        }
        gpu_size_ = 0;
    }

    void Memory::release_cpu() {
        if (cpu_) {
            if(owner_cpu_){
                CudaDeviceGuard guard(device_id_);
                checkCudaRuntime(cudaFreeHost(cpu_));
            }
            cpu_ = nullptr;
        }
        cpu_size_ = 0;
    }

    void Memory::release_all() {
        release_cpu();
        release_gpu();
    }

    void Memory::reference_data(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size) {
        release_all();

        if(cpu == nullptr || cpu_size == 0){
            cpu = nullptr;
            cpu_size = 0;
        }

        if(gpu == nullptr || gpu_size == 0){
            gpu = nullptr;
            gpu_size = 0;
        }

        cpu_ = cpu;
        cpu_size_ = cpu_size;
        gpu_ = gpu_;
        gpu_size_ = gpu_size;

        owner_cpu_ = !(cpu && cpu_size > 0);
        owner_gpu_ = !(gpu && gpu_size > 0);
        checkCudaRuntime(cudaGetDevice(&device_id_));
    }
    
} // namespace TRT