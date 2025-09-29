/**
 * *****************************************************************************
 * File name:   ac_tensor.h
 * 
 * @brief  自定义host、device内存管理
 * 
 * 
 * Created by Aegon on 2023-04-18
 * Copyright © 2023 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACINFER_ULTRA_TRT_TENSOR_H
#define ACINFER_ULTRA_TRT_TENSOR_H

#include <vector>

#include "trt_memory.h"
struct CUstream_st;

namespace TRT {
    
    typedef CUstream_st CUDAStreamRaw;
    typedef CUDAStreamRaw* CUDAStream;

    typedef struct{ unsigned short _; } float16;

    enum class DataHead : int { 
        Init   = 0,
        Device = 1, // GPU
        Host   = 2  // CPU
    };

    enum class DataType : int {
        Float   = 0,
        Float16 = 1,
        Int32   = 2,
        UInt8   = 3
    };

    float   float16_to_float(float16 value);
    float16 float_to_float16(float value);

    int data_type_size(DataType dt);

    const char* data_head_string(DataHead dh);
    const char* data_type_string(DataType dt);

    class Tensor {
    public:
        Tensor(const Tensor& tensor)                = delete; // 禁拷贝
        Tensor& operator = (const Tensor& tensor)   = delete; // 禁赋值
        
        explicit Tensor(DataType dtype = DataType::Float, std::shared_ptr<Memory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
        explicit Tensor(int N, int C, int H, int W, DataType dtype = DataType::Float, std::shared_ptr<Memory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
        explicit Tensor(int ndims, const int* dims, DataType dtype = DataType::Float, std::shared_ptr<Memory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
        explicit Tensor(const std::vector<int>& dims, DataType dtype = DataType::Float, std::shared_ptr<Memory> data = nullptr, int device_id = CURRENT_DEVICE_ID);

        ~Tensor();

        Tensor& release();

        Tensor& set_to(float value);

        int numel() const;

        inline int      ndims()             const { return shape_.size(); }
        inline int      size(int index)     const { return shape_[index]; }
        inline int      shape(int index)    const { return shape_[index]; }
        inline int      batch()             const { return shape_[0]; }
        inline int      channel()           const { return shape_[1]; }
        inline int      height()            const { return shape_[2]; }
        inline int      width()             const { return shape_[3]; }
        inline DataType type()              const { return dtype_; }
        inline DataHead head()              const { return head_; }

        inline const std::vector<int>&      dims()      const { return shape_; }
        inline const std::vector<size_t>&   strides()   const { return strides_;}

        const char* shape_string()  const { return shape_string_; }
        const char* descriptor()    const;

        bool empty() const;

        int count(int start_axis = 0)   const;
        int device()                    const { return device_id_; }
        
        inline int element_size()           const { return data_type_size(dtype_); }

        inline int bytes()                  const { return bytes_; }
        inline int bytes(int start_axis)    const { return count(start_axis) * element_size(); }

        template<typename ... Args>
        int offset(int index, Args ... index_args) const{
            const int index_array[] = {index, index_args...};
            return offset_array(sizeof...(index_args) + 1, index_array);
        }
        int offset_array(const std::vector<int>& index) const;
        int offset_array(size_t size, const int* index_array) const;

        template<typename ... Args>
        Tensor& resize(int dim_size, Args ... dim_size_args){
            const int dim_size_array[] = {dim_size, dim_size_args...};
            return resize(sizeof...(dim_size_args) + 1, dim_size_array);
        }
        Tensor& resize(int ndims, const int* dims);
        Tensor& resize(const std::vector<int>& dims);
        Tensor& resize_single_dim(int idim, int size);

        std::shared_ptr<Tensor> clone() const;

        Tensor& to_gpu(bool copy=true);
        Tensor& to_cpu(bool copy=true);
        Tensor& to_half();
        Tensor& to_float();

        inline void* cpu() const { ((Tensor*)this)->to_cpu(); return data_->cpu(); }
        inline void* gpu() const { ((Tensor*)this)->to_gpu(); return data_->gpu(); }

        template<typename DType> inline const DType* cpu() const { return (DType*)cpu(); }
        template<typename DType> inline const DType* gpu() const { return (DType*)gpu(); }
        template<typename DType> inline DType* cpu()             { return (DType*)cpu(); }
        template<typename DType> inline DType* gpu()             { return (DType*)gpu(); }
        template<typename DType, typename ... _Args> 
        inline DType* cpu(int i, _Args&& ... args) { return cpu<DType>() + offset(i, args...); }
        template<typename DType, typename ... _Args> 
        inline DType* gpu(int i, _Args&& ... args) { return gpu<DType>() + offset(i, args...); }
        template<typename DType, typename ... _Args> 
        inline DType& at(int i, _Args&& ... args) { return *(cpu<DType>() + offset(i, args...)); }

        Tensor& copy_from_gpu(size_t offset, const void* src, size_t num_element, int device_id = CURRENT_DEVICE_ID);
        Tensor& copy_from_cpu(size_t offset, const void* src, size_t num_element);

        void reference_data(const std::vector<int>& shape, void* cpu_data, size_t cpu_size, void* gpu_data, size_t gpu_size, DataType dtype);

        Tensor& synchronize();

        bool        is_stream_owner()   const { return stream_owner_; }
        CUDAStream  get_stream()        const { return stream_; }
        Tensor&     set_stream(CUDAStream stream, bool owner=false);

        std::shared_ptr<Memory> get_data()             const {return data_;}
        std::shared_ptr<Memory> get_workspace()        const {return workspace_;}
        Tensor& set_workspace(std::shared_ptr<Memory> workspace);

    private:
        void setup_data(std::shared_ptr<Memory> data);
        Tensor& adjust_memory();
        Tensor& compute_shape_string();

    private:
        std::vector<int>    shape_;
        std::vector<size_t> strides_;
        int                 device_id_      = 0;
        size_t              bytes_          = 0;
        DataHead            head_           = DataHead::Init;
        DataType            dtype_          = DataType::Float;
        CUDAStream          stream_         = nullptr;
        bool                stream_owner_   = false;

        char shape_string_[100];
        char descriptor_string_[100];

        std::shared_ptr<Memory> data_;
        std::shared_ptr<Memory> workspace_;
    };

} // namespace TRT


#endif // ACINFER_ULTRA_TRT_TENSOR_H