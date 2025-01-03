
#ifndef ACINFER_ULTRA_TRT_MEMORY_H
#define ACINFER_ULTRA_TRT_MEMORY_H

#include <memory>

#define CURRENT_DEVICE_ID -1

namespace TRT {

    class TRTMemory {
    public:

        TRTMemory(int device_id = CURRENT_DEVICE_ID);
        TRTMemory(void* cpu, size_t cpu_size, void* gpu, size_t gpu_size);

        ~TRTMemory();

        void* gpu(size_t size);
        void* cpu(size_t size);

        void release_gpu();
        void release_cpu();
        void release_all();

        inline bool owner_gpu() const{return owner_gpu_;}
        inline bool owner_cpu() const{return owner_cpu_;}

        inline size_t cpu_size() const{return cpu_size_;}
        inline size_t gpu_size() const{return gpu_size_;}

        inline int device_id() const{return device_id_;}

        inline void* gpu() const { return gpu_; }
        inline void* cpu() const { return cpu_; }

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