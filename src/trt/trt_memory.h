/**
 * *****************************************************************************
 * File name:   ac_memory.h
 * 
 * @brief 自定义内存管理类
 * 
 * 
 * Created by Aegon on 2023-04-18
 * Copyright © 2023 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACINFER_ULTRA_TRT_MEMORY_H
#define ACINFER_ULTRA_TRT_MEMORY_H

#include <memory>

#define CURRENT_DEVICE_ID -1

namespace TRT {

    class CudaDeviceGuard {
    public:
        CudaDeviceGuard(int device_id);
        ~CudaDeviceGuard();
    private:
        int old_device_;
    };
    
    class Memory {
    public:
        Memory(int device_id = CURRENT_DEVICE_ID);
        Memory(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size);

        ~Memory();

        void* cpu(size_t size);
        void* gpu(size_t size);
        
        void release_gpu();
        void release_cpu();
        void release_all();

        inline bool owner_gpu() const{return owner_gpu_;}
        inline bool owner_cpu() const{return owner_cpu_;}

        inline size_t cpu_size() const{return cpu_size_;}
        inline size_t gpu_size() const{return gpu_size_;}

        inline int device_id() const{return device_id_;}

        inline void* cpu() { return cpu_; }
        inline void* gpu() { return gpu_; }

        void reference_data(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size);

    private:
        void* cpu_ = nullptr;
        void* gpu_ = nullptr;

        size_t cpu_size_ = 0;
        size_t gpu_size_ = 0;

        bool owner_cpu_ = true;
        bool owner_gpu_ = true;

        int device_id_ = 0;
    };
    
} // namespace TRT

#endif // ACINFER_ULTRA_TRT_MEMORY_H