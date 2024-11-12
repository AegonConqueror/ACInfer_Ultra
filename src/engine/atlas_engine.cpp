
#include "atlas_engine.h"

inline std::string data_type_string(aclDataType dt){
    switch(dt){
        case ACL_FLOAT:   return "Float";
        case ACL_FLOAT16: return "Float16";
        case ACL_UINT8:   return "UInt8";
        case ACL_INT8:    return "Int8";
        case ACL_INT32:   return "Int32";
        case ACL_INT64:   return "Int64";
        default: return "Unknow";
    }
}

AtlasEngine::AtlasEngine() 
    : deviceId_(0), context_(nullptr), stream_(nullptr), modelId_(0), modelWorkSize_(0), 
    modelWeightSize_(0), modelWorkPtr_(nullptr), modelWeightPtr_(nullptr), loadFlag_(false), 
    modelDesc_(nullptr), input_(nullptr), output_(nullptr), is_init_(true), isDevice_(false) { }

error_e AtlasEngine::Destory() {
    if (loadFlag_) {
        aclError ret = aclmdlUnload(modelId_);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("unload model failed, modelId is %u, errorCode is %d",
                modelId_, static_cast<int32_t>(ret));
        }

        if (modelDesc_ != nullptr) {
            (void)aclmdlDestroyDesc(modelDesc_);
            modelDesc_ = nullptr;
        }

        if (modelWorkPtr_ != nullptr) {
            (void)aclrtFree(modelWorkPtr_);
            modelWorkPtr_ = nullptr;
            modelWorkSize_ = 0;
        }

        if (modelWeightPtr_ != nullptr) {
            (void)aclrtFree(modelWeightPtr_);
            modelWeightPtr_ = nullptr;
            modelWeightSize_ = 0;
        }

        loadFlag_ = false;
        LOG_INFO("unload model success, modelId is %u", modelId_);
        modelId_ = 0;
    }

    if (input_ != nullptr) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input_); ++i) {
            aclDataBuffer *dataBuffer = aclmdlGetDatasetBuffer(input_, i);
            (void)aclDestroyDataBuffer(dataBuffer);
        }
        (void)aclmdlDestroyDataset(input_);
        input_ = nullptr;

        for (auto inputBuffer : inputBufferList_) {
            aclrtFree(inputBuffer);
            inputBuffer = nullptr;
        }
        inputBufferList_.clear();
        LOG_INFO("destroy model input success");
    }

    if (output_ != nullptr) {
        for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output_); ++i) {
            aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
            void* data = aclGetDataBufferAddr(dataBuffer);
            (void)aclrtFree(data);
            (void)aclDestroyDataBuffer(dataBuffer);
        }

        (void)aclmdlDestroyDataset(output_);
        output_ = nullptr;
        LOG_INFO("destroy model output success");
    }
    
    if (owner_device_) {
        aclError ret;
        if (stream_ != nullptr) {
            ret = aclrtDestroyStream(stream_);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("destroy stream failed, errorCode = %d", static_cast<int32_t>(ret));
            }
            stream_ = nullptr;
        }
        LOG_INFO("end to destroy stream");

        if (context_ != nullptr) {
            ret = aclrtDestroyContext(context_);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("destroy context failed, errorCode = %d", static_cast<int32_t>(ret));
            }
            context_ = nullptr;
        }
        LOG_INFO("end to destroy context");

        ret = aclrtResetDevice(deviceId_);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("reset device %d failed, errorCode = %d", deviceId_, static_cast<int32_t>(ret));
        }
        LOG_INFO("end to reset device %d", deviceId_);

        ret = aclFinalize();
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("finalize acl failed, errorCode = %d", static_cast<int32_t>(ret));
        }
        LOG_INFO("end to finalize acl");
    }
}

void AtlasEngine::Print() {
    LOG_INFO("****************************************************************************");
    LOG_INFO("Infer %p detail", this);
    size_t inputCount = aclmdlGetNumInputs(modelDesc_);
    LOG_INFO("\tInputs: %d", inputCount);
    for (size_t i = 0; i < inputCount; i++) {
        auto name = aclmdlGetInputNameByIndex(modelDesc_, i);
        auto input_type = aclmdlGetInputDataType(modelDesc_, i);
        aclmdlIODims intput_dim;
        aclmdlGetInputDims(modelDesc_, i, &intput_dim);
        std::vector<int64_t> dim_vec(intput_dim.dims, intput_dim.dims + intput_dim.dimCount);
        LOG_INFO("\t\t%d.%-24s : shape {%s} %s", i, name, iTools::vector_shape_string(dim_vec).c_str(), data_type_string(input_type).c_str());
    }

    size_t outputCount = aclmdlGetNumOutputs(modelDesc_);
    LOG_INFO("\tOutputs: %d", outputCount);
    for (size_t i = 0; i < outputCount; i++) {
        auto name = aclmdlGetOutputNameByIndex(modelDesc_, i);
        auto output_type = aclmdlGetOutputDataType(modelDesc_, i);
        aclmdlIODims output_dim;
        aclmdlGetOutputDims(modelDesc_, i, &output_dim);
        std::vector<int64_t> dim_vec(output_dim.dims, output_dim.dims + output_dim.dimCount);
        LOG_INFO("\t\t%d.%-24s : shape {%s} %s", i, name, iTools::vector_shape_string(dim_vec).c_str(), data_type_string(output_type).c_str());
    }
    LOG_INFO("****************************************************************************");
}

std::vector<int> AtlasEngine::GetInputShape(int index) {
    aclmdlIODims inputdims;
    aclmdlGetInputDims(modelDesc_, index, &inputdims);
    std::vector<int> dim_vec(inputdims.dims, inputdims.dims + inputdims.dimCount);
    return dim_vec;
}

std::vector<std::vector<int>> AtlasEngine::GetOutputShapes() {
    size_t outputCount = aclmdlGetNumOutputs(modelDesc_);
    std::vector<std::vector<int>> output_shapes;
    for (size_t i = 0; i < outputCount; i++) {
        aclmdlIODims output_dim;
        aclmdlGetOutputDims(modelDesc_, i, &output_dim);
        std::vector<int> dim_vec(output_dim.dims, output_dim.dims + output_dim.dimCount);
        output_shapes.push_back(dim_vec);
    }
    return output_shapes;
}

std::string AtlasEngine::GetInputType(int index) {
    auto input_type = aclmdlGetInputDataType(modelDesc_, index);
    return data_type_string(input_type);
}

std::vector<std::string> AtlasEngine::GetOutputTypes() {
    size_t outputCount = aclmdlGetNumOutputs(modelDesc_);
    std::vector<std::string> output_chars;
    for (size_t i = 0; i < outputCount; i++) {
        auto output_type = aclmdlGetOutputDataType(modelDesc_, i);
        output_chars.push_back(data_type_string(output_type));
    }
    return output_chars;
}

error_e AtlasEngine::Initialize(const std::string &file, bool owner_device, bool use_plugins) {
    owner_device_ = owner_device;
    aclError ret;
    if (owner_device_) {
        // /*          ---------- ACL初始化 ----------          */
        ret = aclInit(nullptr);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("acl init failed, errorCode = %d", static_cast<int32_t>(ret));
            return DEVICE_INIT_FAIL;
        }
        LOG_INFO("acl init success");

        // 1. 指定运算的Device
        ret = aclrtSetDevice(deviceId_);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("acl set device %d failed, errorCode = %d", deviceId_, static_cast<int32_t>(ret));
            return DEVICE_INIT_FAIL;
        }
        LOG_INFO("set device %d success", deviceId_);

        // 2. 显式创建一个Context，用于管理Stream对象
        ret = aclrtCreateContext(&context_, deviceId_);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("acl create context failed, deviceId = %d, errorCode = %d",
                deviceId_, static_cast<int32_t>(ret));
            return DEVICE_INIT_FAIL;
        }
        LOG_INFO("create context success");

        // 3. 显式创建一个Stream，用于维护一些异步操作的执行顺序，确保按照应用程序中的代码调用顺序执行任务
        ret = aclrtCreateStream(&stream_);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("acl create stream failed, deviceId = %d, errorCode = %d",
                deviceId_, static_cast<int32_t>(ret));
            return DEVICE_INIT_FAIL;
        }
        LOG_INFO("create stream success");

        // get run mode
        // runMode is ACL_HOST which represents app is running in host
        // runMode is ACL_DEVICE which represents app is running in device
        aclrtRunMode runMode;
        ret = aclrtGetRunMode(&runMode);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("acl get run mode failed, errorCode = %d", static_cast<int32_t>(ret));
            return DEVICE_INIT_FAIL;
        }
        isDevice_ = (runMode == ACL_DEVICE);
        LOG_INFO("get run mode success");
    }
    
    // /*          ---------- 模型初始化 ----------          */
    if (loadFlag_){
        LOG_ERROR("model has already been loaded");
        return LOAD_MODEL_FAIL;
    }

    // 1.根据om模型文件获取模型执行时所需的权值内存大小、工作内存大小。
    ret = aclmdlQuerySize(model_file.c_str(), &modelWorkSize_, &modelWeightSize_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("query model failed, model file is %s, errorCode is %d",
            model_file.c_str(), static_cast<int32_t>(ret));
        return LOAD_MODEL_FAIL;
    }

    // 2.根据工作内存大小，申请Device上模型执行的工作内存。
    ret = aclrtMalloc(&modelWorkPtr_, modelWorkSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("malloc buffer for work failed, require size is %zu, errorCode is %d",
            modelWorkSize_, static_cast<int32_t>(ret));
        return LOAD_MODEL_FAIL;
    }

    // 3.根据权值内存的大小，申请Device上模型执行的权值内存。
    ret = aclrtMalloc(&modelWeightPtr_, modelWeightSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("malloc buffer for weight failed, require size is %zu, errorCode is %d",
            modelWeightSize_, static_cast<int32_t>(ret));
        return LOAD_MODEL_FAIL;
    }

    // 4.以从om模型文件加载模型、由用户管理工作内存和权值内存
    ret = aclmdlLoadFromFileWithMem(model_file.c_str(), &modelId_, modelWorkPtr_,
        modelWorkSize_, modelWeightPtr_, modelWeightSize_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("load model from file failed, model file is %s, errorCode is %d",
            model_file.c_str(), static_cast<int32_t>(ret));
        return LOAD_MODEL_FAIL;
    }

    loadFlag_ = true;
    LOG_INFO("load model %s success", model_file.c_str());

    // /*          ---------- 获取模型的描述信息 ----------          */
    modelDesc_ = aclmdlCreateDesc();
    ret = aclmdlGetDesc(modelDesc_, modelId_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("get model description failed, modelId is %u, errorCode is %d",
            modelId_, static_cast<int32_t>(ret));
        return LOAD_MODEL_FAIL;
    }
    LOG_INFO("create model description success");

    // /*          ---------- 准备模型推理的输入数据结构 ----------          */
    input_ = aclmdlCreateDataset();
    size_t inputSize = aclmdlGetNumInputs(modelDesc_);
    for (size_t i = 0; i < inputSize; i++) {
        size_t modelInputSize = aclmdlGetInputSizeByIndex(modelDesc_, i);
        void *inputBuffer = nullptr;
        ret = aclrtMalloc(&inputBuffer, modelInputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("malloc device buffer failed. size is %zu, errorCode is %d",
                modelInputSize, static_cast<int32_t>(ret));
            return INPUT_ATTR_ERROR;
        }
        
        aclDataBuffer *inputData = aclCreateDataBuffer(inputBuffer, modelInputSize);
        ret = aclmdlAddDatasetBuffer(input_, inputData);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("add input dataset buffer failed, errorCode is %d", static_cast<int32_t>(ret));
            (void)aclDestroyDataBuffer(inputData);
            inputData = nullptr;
            return INPUT_ATTR_ERROR;
        }
        inputBufferList_.push_back(inputBuffer);
    }
    LOG_INFO("create model input success");

    // /*          ---------- 准备模型推理的输出数据结构 ----------          */
    output_ = aclmdlCreateDataset();
    size_t outputSize = aclmdlGetNumOutputs(modelDesc_);
    for (size_t i = 0; i < outputSize; ++i) {
        size_t modelOutputSize = aclmdlGetOutputSizeByIndex(modelDesc_, i);
        void *outputBuffer = nullptr;
        ret = aclrtMalloc(&outputBuffer, modelOutputSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("can't malloc buffer, size is %zu, create output failed, errorCode is %d",
                modelOutputSize, static_cast<int32_t>(ret));
            return OUTPUT_ATTR_ERROR;
        }

        aclDataBuffer *outputData = aclCreateDataBuffer(outputBuffer, modelOutputSize);
        if (outputData == nullptr) {
            LOG_ERROR("can't create data buffer, create output failed");
            (void)aclrtFree(outputBuffer);
            return OUTPUT_ATTR_ERROR;
        }

        ret = aclmdlAddDatasetBuffer(output_, outputData);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("can't add data buffer, create output failed, errorCode is %d",
                static_cast<int32_t>(ret));
            (void)aclrtFree(outputBuffer);
            (void)aclDestroyDataBuffer(outputData);
            return OUTPUT_ATTR_ERROR;
        }
    }
    LOG_INFO("create model output success");

    return SUCCESS;
}

void AtlasEngine::BindingInput(InferenceDataType& inputData) {
    if (inputData.size() != inputBufferList_.size()) {
        LOG_ERROR("inputs num not match! inputData.size()=%d, inputBufferList.size()=%d", inputData.size(), inputBufferList_.size());
        return;
    }

    aclError aclRet;
    if (context_ == nullptr) {
        LOG_ERROR("context is null");
        return;
    }

    for (size_t i = 0; i < inputData.size(); i++) {
        auto inputBuff = inputData[i].first;
        auto inputDataSize = inputData[i].second;
        auto inputBufferSize = aclmdlGetInputSizeByIndex(modelDesc_, i);
        if (inputDataSize != inputBufferSize) {
            LOG_ERROR("input size not match! inputDataSize=%d, inputBufferSize=%d", inputDataSize, inputBufferSize);
            return;
        }
        if (!isDevice_) {
            aclRet = aclrtMemcpy(inputBufferList_[i], inputBufferSize, inputBuff, inputBufferSize, ACL_MEMCPY_HOST_TO_DEVICE);
            if (aclRet != ACL_SUCCESS) {
                LOG_ERROR("memcpy failed. buffer size is %zu, errorCode is %d", inputBufferSize, static_cast<int32_t>(aclRet));
                return;
            }
        } else {
            aclRet = aclrtMemcpy(inputBufferList_[i], inputBufferSize, inputBuff, inputBufferSize, ACL_MEMCPY_DEVICE_TO_DEVICE);
            if (aclRet != ACL_SUCCESS) {
                LOG_ERROR("memcpy failed. buffer size is %zu, errorCode is %d", inputBufferSize, static_cast<int32_t>(aclRet));
                return;
            }
        }
    }
}

void AtlasEngine::GetInferOutput(InferenceDataType& outputData) {
    auto pred_num = aclmdlGetDatasetNumBuffers(output_);
    outputData.reserve(pred_num);
    for (size_t i = 0; i < pred_num; ++i) {
        // get model output data
        aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(output_, i);
        void* data = aclGetDataBufferAddr(dataBuffer);
        uint32_t len = aclGetDataBufferSizeV2(dataBuffer);

        void *outHostData = nullptr;
        aclError ret = ACL_SUCCESS;
        void *outData = nullptr;
        if (!isDevice_) {
            aclError ret = aclrtMallocHost(&outHostData, len);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("aclrtMallocHost failed, malloc len[%u], errorCode[%d]",
                    len, static_cast<int32_t>(ret));
                return;
            }

            ret = aclrtMemcpy(outHostData, len, data, len, ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("aclrtMemcpy failed, errorCode[%d]", static_cast<int32_t>(ret));
                (void)aclrtFreeHost(outHostData);
                return;
            }
            outData = reinterpret_cast<void*>(outHostData);
        } else {
            outData = reinterpret_cast<void*>(data);
        }
        outputData.emplace_back(std::make_pair(outData, len));

        if (!isDevice_) {
            ret = aclrtFreeHost(outHostData);
            if (ret != ACL_SUCCESS) {
                LOG_ERROR("aclrtFreeHost failed, errorCode[%d]", static_cast<int32_t>(ret));
                return;
            }
        }
    }
    return;
}