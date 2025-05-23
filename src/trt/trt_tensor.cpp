
#include "trt_tensor.h"

#include <cuda_fp16.h>

#include "utils/utils.h"
#include "trt_cuda.h"

namespace TRT {

    float float16_to_float(float16 value){
		return __half2float(*reinterpret_cast<__half*>(&value));
	}

	float16 float_to_float16(float value){
		auto val = __float2half(value);
		return *reinterpret_cast<float16*>(&val);
	}

    inline static int get_device(int device_id){
		if(device_id != CURRENT_DEVICE_ID){
			iCUDA::check_device_id(device_id);
			return device_id;
		}

		checkCudaRuntime(cudaGetDevice(&device_id));
		return device_id;
	}

    /**
     * @brief  封装 sizeof()
     */
    int data_type_size(DataType dt){
		switch (dt) {
			case DataType::Float: 	return sizeof(float);
			case DataType::Float16: return sizeof(float16);
			case DataType::Int32: 	return sizeof(int);
			case DataType::UInt8: 	return sizeof(uint8_t);
			default: {
				LOG_ERROR("Not support dtype: %d", dt);
				return -1;
			}
		}
	}

    /**
     * @brief  数据状态描述字符串
     */
    const char* data_head_string(DataHead dh){
		switch(dh){
			case DataHead::Init:    return "Init";
			case DataHead::Device:  return "Device";
			case DataHead::Host:    return "Host";
			default: return "Unknow";
		}
	}

    /**
     * @brief  数据类型描述字符串
     */
	const char* data_type_string(DataType dt){
		switch(dt){
			case DataType::Float:   return "Float32";
			case DataType::Float16: return "Float16";
			case DataType::Int32:   return "Int32";
			case DataType::UInt8:   return "UInt8";
			default: return "Unknow";
		}
	}

    Tensor::Tensor(DataType dtype, std::shared_ptr<TRTMemory> data, int device_id) {
        shape_string_[0] = 0;
		descriptor_string_[0] = 0;
		this->device_id_ = get_device(device_id);
		this->dtype_ = dtype;
		setup_data(data);
    }

    Tensor::Tensor(int n, int c, int h, int w, DataType dtype, std::shared_ptr<TRTMemory> data, int device_id) {
        this->dtype_ = dtype;
		this->device_id_ = get_device(device_id);
		descriptor_string_[0] = 0;
		setup_data(data);
		resize(n, c, h, w);
    }

    Tensor::Tensor(int ndims, const int* dims, DataType dtype, std::shared_ptr<TRTMemory> data, int device_id) {
        this->dtype_ = dtype;
		this->device_id_ = get_device(device_id);
		descriptor_string_[0] = 0;
		setup_data(data);
		resize(ndims, dims);
    }

    Tensor::Tensor(const std::vector<int>& dims, DataType dtype, std::shared_ptr<TRTMemory> data, int device_id) {
        this->dtype_ = dtype;
		this->device_id_ = get_device(device_id);
		descriptor_string_[0] = 0;
		setup_data(data);
		resize(dims);
    }

    template<typename _T>
	static inline void memset_any_type(_T* ptr, size_t count, _T value){
		for (size_t i = 0; i < count; ++i)
			*ptr++ = value;
	}

    Tensor& Tensor::set_to(float value) {
		int c = count();
		if (dtype_ == DataType::Float) {
			memset_any_type(cpu<float>(), c, value);
		}
		else if(dtype_ == DataType::Float16) {
			memset_any_type(cpu<float16>(), c, float_to_float16(value));
		}
		else if(dtype_ == DataType::Int32) {
			memset_any_type(cpu<int>(), c, (int)value);
		}
		else if(dtype_ == DataType::UInt8) {
			memset_any_type(cpu<uint8_t>(), c, (uint8_t)value);
		}
		else{
			LOG_ERROR("Unsupport type: %d", dtype_);
		}
		return *this;
	}

    Tensor::~Tensor() {
		release();
	}

    Tensor& Tensor::release() {
		data_->release_all();
		shape_.clear();
		bytes_ = 0;
		head_ = DataHead::Init;
		if(stream_owner_ && stream_ != nullptr){
			iCUDA::AutoDevice auto_device_exchange(this->device());
			checkCudaRuntime(cudaStreamDestroy(stream_));
		}
		stream_owner_ = false;
		stream_ = nullptr;
		return *this;
	}

    const char* Tensor::descriptor() const{
		char* descriptor_ptr = (char*)descriptor_string_;
		int device_id = device();
		snprintf(descriptor_ptr, sizeof(descriptor_string_), 
			"Tensor:%p, %s, %s, CUDA:%d", 
			data_.get(),
			data_type_string(dtype_), 
			shape_string_, 
			device_id
		);
		return descriptor_ptr;
	}

    /**
     * @brief  计算 Tensor中元素的总数
     */
    int Tensor::numel() const{
		int value = shape_.empty() ? 0 : 1;
		for(int i = 0; i < shape_.size(); ++i){
			value *= shape_[i];
		}
		return value;
	}

    void Tensor::setup_data(std::shared_ptr<TRTMemory> data) {
        data_ = data;
		if(data_ == nullptr) {
			data_ = std::make_shared<TRTMemory>(device_id_);
		} else {
			device_id_ = data_->device_id();
		}

		head_ = DataHead::Init;
		if(data_->cpu()) {
			head_ = DataHead::Host;
		}

		if(data_->gpu()) {
			head_ = DataHead::Device;
		}
    }

    Tensor& Tensor::adjust_memory() {
		int needed_size = this->numel() * element_size();
		if(needed_size > this->bytes_){
			head_ = DataHead::Init;
		}
		this->bytes_ = needed_size;
		return *this;
	}

    Tensor& Tensor::compute_shape_string(){
		shape_string_[0] = 0;

		char* buffer = shape_string_;
		size_t buffer_size = sizeof(shape_string_);
		for(int i = 0; i < shape_.size(); ++i){

			int size = 0;
			if(i < shape_.size() - 1)
				size = snprintf(buffer, buffer_size, "%d x ", shape_[i]);
			else
				size = snprintf(buffer, buffer_size, "%d", shape_[i]);

			buffer += size;
			buffer_size -= size;
		}
		return *this;
	}

    /**
     * @brief 计算从指定维度开始到数据最后一个维度之间的维度大小乘积，即元素的数量
     */
    int Tensor::count(int start_axis) const {
		if(start_axis >= 0 && start_axis < shape_.size()){
			int size = 1;
			for (int i = start_axis; i < shape_.size(); ++i) 
				size *= shape_[i];
			return size;
		}else{
			return 0;
		}
	}

    bool Tensor::empty() const{
		return data_->cpu() == nullptr && data_->gpu() == nullptr;
	}

    /**
     * @brief 计算给定索引数组所表示的元素在张量中的偏移量
     */
    int Tensor::offset_array(const std::vector<int>& index_array) const{
		return offset_array(index_array.size(), index_array.data());
	}

    /**
     * @brief 计算给定索引数组所表示的元素在张量中的偏移量
     */
    int Tensor::offset_array(size_t size, const int* index_array) const{
		Assert(size <= shape_.size());
		int value = 0;
		for(int i = 0; i < shape_.size(); ++i){

			if(i < size)
				value += index_array[i];

			if(i + 1 < shape_.size())
				value *= shape_[i+1];
		}
		return value;
	}

    /**
     * @brief  重新调整张量的维度和形状，并重新计算张量的步幅（strides）以及相应的内存布局
     */
    Tensor& Tensor::resize(const std::vector<int>& dims) {
		return resize(dims.size(), dims.data());
	}

    /**
     * @brief  重新调整张量的维度和形状，并重新计算张量的步幅（strides）以及相应的内存布局
     */
    Tensor& Tensor::resize(int ndims, const int* dims) {
		std::vector<int> setup_dims(ndims);
		for(int i = 0; i < ndims; ++i){
			int dim = dims[i];
			if(dim == -1){
				Assert(ndims == shape_.size());
				dim = shape_[i];
			}
			setup_dims[i] = dim;
		}
		this->shape_ = setup_dims;

		this->strides_.resize(setup_dims.size());
		
		size_t prev_size  = element_size();
		size_t prev_shape = 1;
		for(int i = (int)strides_.size() - 1; i >= 0; --i){
			if(i + 1 < strides_.size()){
				prev_size  = strides_[i+1];
				prev_shape = shape_[i+1];
			}
			strides_[i] = prev_size * prev_shape;
		}

		this->adjust_memory();
		this->compute_shape_string();
		return *this;
	}

    /**
     * @brief  重置单个维度的维度值
     */
    Tensor& Tensor::resize_single_dim(int idim, int size){
		Assert(idim >= 0 && idim < shape_.size());
		auto new_shape = shape_;
		new_shape[idim] = size;
		return resize(new_shape);
	}

    std::shared_ptr<Tensor> Tensor::clone() const{
		auto new_tensor = std::make_shared<Tensor>(shape_, dtype_);
		if(head_ == DataHead::Init)
			return new_tensor;
		
		if(head_ == DataHead::Host){
			memcpy(new_tensor->cpu(), this->cpu(), this->bytes_);
		}else if(head_ == DataHead::Device){
			iCUDA::AutoDevice auto_device_exchange(device());
			checkCudaRuntime(cudaMemcpyAsync(new_tensor->gpu(), this->gpu(), bytes_, cudaMemcpyDeviceToDevice, stream_));
		}
		return new_tensor;
	}

    Tensor& Tensor::to_gpu(bool copy) {
		if (head_ == DataHead::Device)
			return *this;

		head_ = DataHead::Device;
		data_->gpu(bytes_);

		if (copy && data_->cpu() != nullptr) {
			iCUDA::AutoDevice auto_device_exchange(this->device());
			checkCudaRuntime(cudaMemcpyAsync(data_->gpu(), data_->cpu(), bytes_, cudaMemcpyHostToDevice, stream_));
		}
		return *this;
	}

    Tensor& Tensor::to_cpu(bool copy) {
		if (head_ == DataHead::Host)
			return *this;

		head_ = DataHead::Host;
		data_->cpu(bytes_);

		if (copy && data_->gpu() != nullptr) {
			iCUDA::AutoDevice auto_device_exchange(this->device());
			checkCudaRuntime(cudaMemcpyAsync(data_->cpu(), data_->gpu(), bytes_, cudaMemcpyDeviceToHost, stream_));
			checkCudaRuntime(cudaStreamSynchronize(stream_));
		}
		return *this;
	}

    Tensor& Tensor::to_float() {
		if (type() == DataType::Float)
			return *this;

		if (type() != DataType::Float16) {
			LOG_INFO("not implement function");
			return *this;
		}

		auto c = count();
		float* convert_memory = (float*)malloc(c * data_type_size(DataType::Float));
		float* dst = convert_memory;
		float16* src = cpu<float16>();

		for (int i = 0; i < c; ++i)
			*dst++ = float16_to_float(*src++);

		this->dtype_ = DataType::Float;
		adjust_memory();
		memcpy(cpu(), convert_memory, bytes_);
		free(convert_memory);
		return *this;
	}

    Tensor& Tensor::to_half() {
		if (type() == DataType::Float16)
			return *this;

		if (type() != DataType::Float) {
			LOG_INFO("not implement function");
			return *this;
		}

		auto c = count();
		float16* convert_memory = (float16*)malloc(c * data_type_size(DataType::Float16));
		float16* dst = convert_memory;
		float* src = cpu<float>();

		for (int i = 0; i < c; ++i) 
			*dst++ = float_to_float16(*src++);

		this->dtype_ = DataType::Float16;
		adjust_memory();
		memcpy(cpu(), convert_memory, bytes_);
		free(convert_memory);
		return *this;
	}

    /**
     * @brief 从GPU内存中的源数据复制元素到张量对象所管理的GPU内存中
     * 
     * @param offset [偏移量]
     * @param src [gpu数据指针]
     * @param num_element [拷贝元素总个数]
     * @param device_id [gpu id]
     * @return Tensor& [Tensor]
     * @note [修改详情]
     */
    Tensor& Tensor::copy_from_gpu(size_t offset, const void* src, size_t num_element, int device_id){
		if(head_ == DataHead::Init)
			to_gpu(false);

		size_t offset_location = offset * element_size();
		if(offset_location >= bytes_){
			LOG_ERROR("Offset location[%lld] >= bytes_[%lld], out of range", offset_location, bytes_);
			return *this;
		}

		size_t copyed_bytes = num_element * element_size();
		size_t remain_bytes = bytes_ - offset_location;
		if(copyed_bytes > remain_bytes){
			LOG_ERROR("Copyed bytes[%lld] > remain bytes[%lld], out of range", copyed_bytes, remain_bytes);
			return *this;
		}
		
		if(head_ == DataHead::Device){
			int current_device_id = get_device(device_id);
			int gpu_device_id = device();
			if(current_device_id != gpu_device_id){
				checkCudaRuntime(cudaMemcpyPeerAsync(gpu<unsigned char>() + offset_location, gpu_device_id, src, current_device_id, copyed_bytes, stream_));
			}
			else{
				checkCudaRuntime(cudaMemcpyAsync(gpu<unsigned char>() + offset_location, src, copyed_bytes, cudaMemcpyDeviceToDevice, stream_));
			}
		}else if(head_ == DataHead::Host){
			iCUDA::AutoDevice auto_device_exchange(this->device());
			checkCudaRuntime(cudaMemcpyAsync(cpu<unsigned char>() + offset_location, src, copyed_bytes, cudaMemcpyDeviceToHost, stream_));
		}else{
			LOG_ERROR("Unsupport head type %d", head_);
		}
		return *this;
	}

    /**
     * @brief 从CPU内存中的源数据复制元素到张量对象所管理的GPU内存中
     * 
     * @param offset [偏移量]
     * @param src [cpu数据指针]
     * @param num_element [拷贝元素总个数]
     * @return Tensor& [Tensor]
     * @note [修改详情]
     */
    Tensor& Tensor::copy_from_cpu(size_t offset, const void* src, size_t num_element){
		if(head_ == DataHead::Init)
			to_cpu(false);

		size_t offset_location = offset * element_size();
		if(offset_location >= bytes_){
			LOG_ERROR("Offset location[%lld] >= bytes_[%lld], out of range", offset_location, bytes_);
			return *this;
		}

		size_t copyed_bytes = num_element * element_size();
		size_t remain_bytes = bytes_ - offset_location;
		if(copyed_bytes > remain_bytes){
			LOG_ERROR("Copyed bytes[%lld] > remain bytes[%lld], out of range", copyed_bytes, remain_bytes);
			return *this;
		}

		if(head_ == DataHead::Device){
			iCUDA::AutoDevice auto_device_exchange(this->device());
			checkCudaRuntime(cudaMemcpyAsync((char*)data_->gpu() + offset_location, src, copyed_bytes, cudaMemcpyHostToDevice, stream_));
		}else if(head_ == DataHead::Host){
			memcpy((char*)data_->cpu() + offset_location, src, copyed_bytes);
		}else{
			LOG_ERROR("Unsupport head type %d", head_);
		}
		return *this;
	}
    
    void Tensor::reference_data(const std::vector<int>& shape, void* cpu_data, size_t cpu_size, void* gpu_data, size_t gpu_size, DataType dtype){
		dtype_ = dtype;
		data_->reference_data(cpu_data, cpu_size, gpu_data, gpu_size);
		setup_data(data_);
		resize(shape);
	}

    Tensor& Tensor::synchronize(){ 
		iCUDA::AutoDevice auto_device_exchange(this->device());
		checkCudaRuntime(cudaStreamSynchronize(stream_));
		return *this;
	}

    Tensor& Tensor::set_stream(CUDAStream stream, bool owner) {
        stream_ = stream; 
        stream_owner_ = owner; 
        return *this;
    }

    Tensor& Tensor::set_workspace(std::shared_ptr<TRTMemory> workspace) { 
        workspace_ = workspace; 
        return *this; 
    }

    /**
     * @brief 设置张量中的某个 batch 的数据为 cv::Mat对象 的数据
     * 
     * @param n [batch的索引]
     * @param image [图像数据]
     */
    Tensor& Tensor::set_mat(int n, const cv::Mat& _image) {
		cv::Mat image = _image;
		Assert(!image.empty() && CV_MAT_DEPTH(image.type()) == CV_32F && type() == DataType::Float);
		Assert(shape_.size() == 4 && n < shape_[0] && image.channels() == shape_[1]);
		to_cpu(false);

		int width  = shape_[3];
		int height = shape_[2];
		if (image.size() != cv::Size(width, height))
			cv::resize(image, image, cv::Size(width, height));

		if (image.channels() == 1) {
			memcpy(cpu<float>(n), image.data, width * height * sizeof(float));
			return *this;
		}

		std::vector<cv::Mat> ms(image.channels());
		for (int i = 0; i < ms.size(); ++i) 
			ms[i] = cv::Mat(height, width, CV_32F, cpu<float>(n, i));

		cv::split(image, &ms[0]);
		Assert((void*)ms[0].data == (void*)cpu<float>(n));
		return *this;
	}

    /**
     * @brief 设置张量中的某个 batch 的数据为 cv::Mat对象 的归一化数据
     * 
     * @param n [batch的索引]
     * @param image [图像数据]
     */
    Tensor& Tensor::set_norm_mat(int n, const cv::Mat& image, float mean[3], float std[3]) {
		Assert(image.channels() == 3 && !image.empty() && type() == DataType::Float);
		Assert(ndims() == 4 && n < shape_[0]);
		to_cpu(false);

		int width   = shape_[3];
		int height  = shape_[2];
		float scale = 1 / 255.0;
		cv::Mat inputframe = image;
		if(inputframe.size() != cv::Size(width, height))
			cv::resize(inputframe, inputframe, cv::Size(width, height));

		if(CV_MAT_DEPTH(inputframe.type()) != CV_32F){
			inputframe.convertTo(inputframe, CV_32F, scale);
		}

		cv::Mat ms[3];
		for (int c = 0; c < 3; ++c)
			ms[c] = cv::Mat(height, width, CV_32F, cpu<float>(n, c));

		split(inputframe, ms);
		Assert((void*)ms[0].data == (void*)cpu<float>(n));

		for (int c = 0; c < 3; ++c)
			ms[c] = (ms[c] - mean[c]) / std[c];
		return *this;
	}
    
} // namespace TRT
