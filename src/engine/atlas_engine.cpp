
#include "engine.h"
#include "acl/acl.h"

static ac_tensor_type_e engine_type_convert(aclDataType type) {
    switch (type)
    {
    case ACL_FLOAT:
        return AC_TENSOR_FLOAT;
    case ACL_FLOAT16:
        return AC_TENSOR_FLOAT16;
    case ACL_UINT8:
        return AC_TENSOR_UINT8;
    case ACL_INT8:
        return AC_TENSOR_INT8;
    case ACL_INT16:
        return AC_TENSOR_INT16;
    case ACL_INT32:
        return AC_TENSOR_INT32;
    case ACL_INT64:
        return AC_TENSOR_INT64;
    default:
        LOG_ERROR("unsupported rknn type: %d\n", type);
        exit(1);
    }
}

static aclDataType engine_type_convert(ac_tensor_type_e type) {
    switch (type)
    {
    case AC_TENSOR_FLOAT:
        return ACL_FLOAT;
    case AC_TENSOR_FLOAT16:
        return ACL_FLOAT16;
    case AC_TENSOR_UINT8:
        return ACL_UINT8;
    case AC_TENSOR_INT8:
        return ACL_INT8;
    case AC_TENSOR_INT16:
        return ACL_INT16;
    case AC_TENSOR_INT32:
        return ACL_INT32;
    case AC_TENSOR_INT64:
        return ACL_INT64;
    default:
        LOG_ERROR("unsupported rknn type: %d\n", type);
        exit(1);
    }
}

static ac_engine_attr engine_tensor_attr_encode(const int index, aclmdlDesc* modelDesc, bool is_input=true) {
    aclmdlIODims intput_dim;

    auto name = is_input ? aclmdlGetInputNameByIndex(modelDesc, index) : aclmdlGetOutputNameByIndex(modelDesc, index);
    auto type = is_input ? aclmdlGetInputDataType(modelDesc, index) : aclmdlGetOutputDataType(modelDesc, index);
    auto modelInputSize = is_input ? aclmdlGetInputSizeByIndex(modelDesc, index) : aclmdlGetOutputSizeByIndex(modelDesc, index);
    auto ret = is_input ? aclmdlGetInputDims(modelDesc, index, &intput_dim) : aclmdlGetOutputDims(modelDesc, index, &intput_dim);

    std::vector<int64_t> dim_vec(intput_dim.dims, intput_dim.dims + intput_dim.dimCount);
    auto n_elems = std::accumulate(std::begin(dim_vec), std::end(dim_vec), 1, std::multiplies<int64_t>());
    
    ac_engine_attr shape;
    shape.index = index;
    shape.name = name;
    shape.n_dims = intput_dim.dimCount;
    for (int i = 0; i < intput_dim.dimCount; ++i) {
        shape.dims[i] = intput_dim.dims[i];
    }

    shape.n_elems = n_elems;
    shape.size = modelInputSize;

    shape.qnt_zp = 0;
    shape.qnt_scale = .0f;

    shape.type = engine_type_convert(type);
    shape.layout = AC_TENSOR_NCHW;

    return shape;
}

class AtlasEngine : public ACEngine {
public:
    AtlasEngine();

    ~AtlasEngine() override { destory(); };

    error_e     create(const std::string &file_path);

    virtual void Print() override;
    virtual void BindingInput(InferenceData& inputData) override;
    virtual void GetInferOutput(InferenceData& outputData, bool sync) override;

    virtual const ac_engine_attrs GetInputAttrs()  override;
    virtual const ac_engine_attrs GetOutputAttrs() override;

    virtual int GetOutputIndex(const std::string name) override;

private:
    error_e init_device();


    void unload_model();
    void destroy_input();
    void destroy_output();
    void destroy_resource();
    void destory_device();

    void destory();

private:
    int                             deviceId_;
    bool                            isDevice_;
    bool                            is_init_;
    bool                            loadFlag_;
    uint32_t                        modelId_;

    aclrtContext 		            context_;
	aclrtStream 			        stream_;

    size_t                          modelWorkSize_;
    size_t                          modelWeightSize_;

    void*                           modelWorkPtr_;
    void*                           modelWeightPtr_;

    aclmdlDesc*                     modelDesc_;
    aclmdlDataset*                  input_;
    aclmdlDataset*                  output_;

    std::vector<void*>              inputBufferList_;

    uint32_t input_num_;
    uint32_t output_num_;

    std::vector<ac_engine_attr> input_attrs_;
    std::vector<ac_engine_attr> output_attrs_;

    std::unordered_map<std::string, uint32_t> name_index_map_;
};

inline std::string data_type_string(ac_tensor_type_e dt){
    switch(dt){
        case AC_TENSOR_FLOAT:   return "Float";
        case AC_TENSOR_FLOAT16: return "Float16";
        case AC_TENSOR_UINT8:   return "UInt8";
        case AC_TENSOR_INT8:    return "Int8";
        case AC_TENSOR_INT16:   return "Int16";
        case AC_TENSOR_INT32:   return "Int32";
        case AC_TENSOR_INT64:   return "Int64";
        default: return "Unknow";
    }
}

inline std::string data_format_string(ac_tensor_fmt_e dt){
    switch(dt){
        case AC_TENSOR_NCHW:   return "NCHW";
        case AC_TENSOR_NHWC:   return "NHWC";
        default: return "Unknow";
    }
}
AtlasEngine::AtlasEngine() : deviceId_(0), context_(nullptr), stream_(nullptr), modelId_(0), 
    modelWorkSize_(0), modelWeightSize_(0), modelWorkPtr_(nullptr), modelWeightPtr_(nullptr), 
    loadFlag_(false), modelDesc_(nullptr), input_(nullptr), output_(nullptr), is_init_(true),
    isDevice_(false) {}

error_e AtlasEngine::init_device(){
    aclError ret;
    ret = aclInit(nullptr);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("acl init failed, errorCode = %d", static_cast<int32_t>(ret));
        return DEVICE_INIT_FAIL;
    }
    LOG_INFO("acl init success");

    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("acl set device 0 failed, errorCode = %d", static_cast<int32_t>(ret));
        return DEVICE_INIT_FAIL;
    }
    LOG_INFO("set device %d success", deviceId_);
    return SUCCESS;
}

void AtlasEngine::unload_model(){
    if (!loadFlag_) {
        LOG_WARNING("no model had been loaded, unload failed");
        return;
    }

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

void AtlasEngine::destroy_input(){
    if (input_ == nullptr) {
        return;
    }

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

void AtlasEngine::destroy_output(){
    if (output_ == nullptr) {
        return;
    }
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

void AtlasEngine::destroy_resource(){
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
}

void AtlasEngine::destory_device() {
    aclError ret;
    ret = aclrtResetDevice(0);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("reset device 0 failed, errorCode = %d", static_cast<int32_t>(ret));
    }
    LOG_INFO("end to reset device 0");

    ret = aclFinalize();
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("finalize acl failed, errorCode = %d", static_cast<int32_t>(ret));
    }
    LOG_INFO("end to finalize acl");
}

void AtlasEngine::destory() {
    unload_model();
    destroy_input();
    destroy_output();
    destroy_resource();
    destory_device();
}

void AtlasEngine::Print() {
    LOG_INFO("****************************************************************************");
    LOG_INFO("Infer %p detail", this);
    LOG_INFO("\tInputs: %d", input_num_);
    for(int i = 0; i < input_num_; ++i){
        auto input_attr = input_attrs_[i];
        std::vector<int64_t> shapes_(input_attr.dims, input_attr.dims + input_attr.n_dims);
        LOG_INFO(
            "\t\t%d.%s : shape {%s}, %s, fmt %s, size %d", 
            i, input_attrs_[i].name.c_str(), 
            iTools::vector_shape_string(shapes_).c_str(), 
            data_type_string(input_attr.type).c_str(),
            data_format_string(input_attr.layout).c_str(),
            input_attr.size
        );
    }

    LOG_INFO("\tOutputs: %d", output_num_);
    for(int i = 0; i < output_num_; ++i){
        auto output_attr = output_attrs_[i];
        std::vector<int64_t> shapes_(output_attr.dims, output_attr.dims + output_attr.n_dims);
        LOG_INFO(
            "\t\t%d.%s : shape {%s}, %s, fmt %s, size %d", 
            i, output_attrs_[i].name.c_str(), 
            iTools::vector_shape_string(shapes_).c_str(),
            data_type_string(output_attr.type).c_str(),
            data_format_string(output_attr.layout).c_str(),
            output_attr.size
        );
    }
    LOG_INFO("****************************************************************************");
}

error_e AtlasEngine::create(const std::string &model_file) {
    if (init_device() != SUCCESS) {
        return DEVICE_INIT_FAIL;
    }
    
    aclError ret;
    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR("acl create context failed, deviceId = %d, errorCode = %d",
            deviceId_, static_cast<int32_t>(ret));
        return DEVICE_INIT_FAIL;
    }
    LOG_INFO("create context success");

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
    input_num_ = aclmdlGetNumInputs(modelDesc_);
    input_attrs_.reserve(input_num_);
    for (size_t i = 0; i < input_num_; i++) {
        input_attrs_.emplace_back(engine_tensor_attr_encode(i, modelDesc_, 1));

        size_t modelInputSize = aclmdlGetInputSizeByIndex(modelDesc_, i);
        void* inputBuffer = nullptr;
        ret = aclrtMalloc(&inputBuffer, modelInputSize, ACL_MEM_MALLOC_NORMAL_ONLY);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("malloc device buffer failed. size is %zu, errorCode is %d",
                modelInputSize, static_cast<int32_t>(ret));
            return INPUT_ATTR_ERROR;
        }
        
        aclDataBuffer* inputData = aclCreateDataBuffer(inputBuffer, modelInputSize);
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
    output_num_ = aclmdlGetNumOutputs(modelDesc_);
    output_attrs_.reserve(output_num_);
    for (size_t i = 0; i < output_num_; ++i) {
        output_attrs_.emplace_back(engine_tensor_attr_encode(i, modelDesc_, 0));

        size_t modelOutputSize = aclmdlGetOutputSizeByIndex(modelDesc_, i);
        void* outputBuffer = nullptr;
        ret = aclrtMalloc(&outputBuffer, modelOutputSize, ACL_MEM_MALLOC_HUGE_FIRST);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR("can't malloc buffer, size is %zu, create output failed, errorCode is %d",
                modelOutputSize, static_cast<int32_t>(ret));
            return OUTPUT_ATTR_ERROR;
        }

        aclDataBuffer* outputData = aclCreateDataBuffer(outputBuffer, modelOutputSize);
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

    for (size_t i = 0; i < output_num_; i++) {
        name_index_map_[output_attrs_[i].name] = i;
    }

    LOG_INFO("create model output success");

    return SUCCESS;
}

void AtlasEngine::BindingInput(InferenceData& inputData) {
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

void AtlasEngine::GetInferOutput(InferenceData& outputData, bool sync) {
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

const ac_engine_attrs AtlasEngine::GetInputAttrs() {
    return input_attrs_;
}

const ac_engine_attrs AtlasEngine::GetOutputAttrs() {
    return output_attrs_;
}

int AtlasEngine::GetOutputIndex(const std::string name) {
    auto it = name_index_map_.find(name);
    if (it != name_index_map_.end()) {
        return it->second;
    } else {
        return -1;
    }
}

std::shared_ptr<ACEngine> create_engine(const std::string &file_path, bool use_plugins) {
    std::shared_ptr<AtlasEngine> Instance(new AtlasEngine());
    if (Instance->create(file_path) != SUCCESS) {
        Instance.reset();
    }
    return Instance;
}