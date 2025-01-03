
#ifndef ACINFER_ULTRA_TRT_TENSOR_H
#define ACINFER_ULTRA_TRT_TENSOR_H

#include <opencv2/opencv.hpp>

#include "trt_memory.h"

struct CUstream_st;
typedef CUstream_st CUDAStreamRaw;

namespace TRT {

    typedef CUDAStreamRaw* CUDAStream;
    typedef struct{ unsigned short _; } float16;
    
    // 数据状态
    enum class DataHead : int { 
        Init   = 0, // 初始
        Device = 1, // 在GPU上
        Host   = 2  // 在CPU上
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
        /**
         * @brief 禁用拷贝构造函数
         * @note [该类的对象不能通过拷贝方式进行构造]
         */
        Tensor(const Tensor& tensor) = delete;

        /**
         * @brief 禁用拷贝赋值运算符
         * @note [该类的对象不能通过拷贝方式进行赋值]
         */
        Tensor& operator = (const Tensor& tensor) = delete;

        explicit Tensor(DataType dtype = DataType::Float, std::shared_ptr<TRTMemory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
        explicit Tensor(int n, int c, int h, int w, DataType dtype = DataType::Float, std::shared_ptr<TRTMemory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
        explicit Tensor(int ndims, const int* dims, DataType dtype = DataType::Float, std::shared_ptr<TRTMemory> data = nullptr, int device_id = CURRENT_DEVICE_ID);
        explicit Tensor(const std::vector<int>& dims, DataType dtype = DataType::Float, std::shared_ptr<TRTMemory> data = nullptr, int device_id = CURRENT_DEVICE_ID);

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

        /**
         * @brief 计算给定索引数组所表示的元素在张量中的偏移量
         */
        template<typename ... Args>
        int offset(Args ... index_args) const{
            const int index_array[] = {index_args...};
            return offset_array(sizeof...(index_args), index_array);
        }
        int offset_array(const std::vector<int>& index) const;
        int offset_array(size_t size, const int* index_array) const;

        /**
         * @brief 重新调整张量的维度和形状，并重新计算张量的步幅（strides）以及相应的内存布局
         */
        template<typename ... Args>
        Tensor& resize(Args ... dim_args) {
            const int dim_size_array[] = {dim_args...};
            return resize(sizeof...(dim_args), dim_size_array);
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

        std::shared_ptr<TRTMemory> get_data()             const {return data_;}
        std::shared_ptr<TRTMemory> get_workspace()        const {return workspace_;}
        Tensor& set_workspace(std::shared_ptr<TRTMemory> workspace);

        Tensor& set_mat     (int n, const cv::Mat& image);
        Tensor& set_norm_mat(int n, const cv::Mat& image, float mean[3], float std[3]);
        cv::Mat at_mat(int n = 0, int c = 0) { return cv::Mat(height(), width(), CV_32F, cpu<float>(n, c)); }
        
    private:
        void setup_data(std::shared_ptr<TRTMemory> data);

        Tensor& adjust_memory();
        Tensor& compute_shape_string();

    private:
        std::vector<int>    shape_;
        std::vector<size_t> strides_;                           // 步幅，表示跨越不同维度时所需要跳过的元素数量
        int                 device_id_      = 0;
        size_t              bytes_          = 0;
        DataHead            head_           = DataHead::Init;
        DataType            dtype_          = DataType::Float;
        CUDAStream          stream_         = nullptr;
        bool                stream_owner_   = false;

        char shape_string_[100];
        char descriptor_string_[100];

        std::shared_ptr<TRTMemory> data_;
        std::shared_ptr<TRTMemory> workspace_;
    };
    
    
} // namespace TRT


#endif // ACINFER_ULTRA_TRT_TENSOR_H