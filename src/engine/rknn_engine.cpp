
#include "engine.h"
#include "types/rknn_type.h"

static unsigned char* load_model(const char *filename, int *model_size) {
    FILE *fp = fopen(filename, "rb");
    if (fp == nullptr) {
        LOG_ERROR("fopen %s fail!", filename);
        return nullptr;
    }
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    unsigned char *model = (unsigned char *)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if (model_len != fread(model, 1, model_len, fp)) {
        LOG_ERROR("fread %s fail!", filename);
        free(model);
        return nullptr;
    }
    *model_size = model_len;
    if (fp) {
        fclose(fp);
    }
    return model;
}

static tensor_attr_s rknn_tensor_attr_convert(const rknn_tensor_attr &attr) {
    tensor_attr_s shape;
    shape.index = attr.index;
    shape.size = attr.size;
    shape.name = attr.name;
    shape.type =  attr.type;
    shape.layout = attr.fmt;
    
    std::vector<int> shapes_(attr.dims, attr.dims+attr.n_dims);
    shape.shape = shapes_;
    return shape;
}

class RKEngine : public ACEngine {
public:
    RKEngine() {};

    ~RKEngine() override { destory(); };

    error_e create(const std::string &file);

    virtual void        Print() override;
    virtual void        BindingInput(InferenceDataType& inputData) override;
    virtual void        GetInferOutput(InferenceDataType& outputData) override;

    virtual std::vector<int>                GetInputShape(int index) override;
    virtual std::vector<std::vector<int>>   GetOutputShapes() override;
    virtual std::string                     GetInputType(int index) override;
    virtual std::vector<std::string>        GetOutputTypes() override;

private:
    error_e destory();

private:
    bool is_int8_;

    rknn_context rknn_ctx_;
    bool ctx_created_;

    uint32_t input_num_;
    uint32_t output_num_;

    std::vector<tensor_attr_s> input_attrs_;
    std::vector<tensor_attr_s> output_attrs_;
    
};

inline std::string data_type_string(rknn_tensor_type dt){
    switch(dt){
        case RKNN_TENSOR_FLOAT32:   return "Float";
        case RKNN_TENSOR_FLOAT16:   return "Float16";
        case RKNN_TENSOR_UINT8:     return "UInt8";
        case RKNN_TENSOR_INT8:      return "Int8";
        case RKNN_TENSOR_INT16:     return "Int16";
        default: return "Unknow";
    }
}

error_e RKEngine::create(const std::string &model_file) {
    int model_len = 0;
    auto model = load_model(model_file.c_str(), &model_len);

    if (model == nullptr) {
        LOG_ERROR("load model file %s fail!", model_file);
        return LOAD_MODEL_FAIL;
    }
    int ret = rknn_init(&rknn_ctx_, model, model_len, 0);
    if (ret < 0) {
        LOG_ERROR("rknn_init fail! ret=%d", ret);
        return DEVICE_INIT_FAIL;
    }
    
    LOG_INFO("rknn_init success!");
    ctx_created_ = true;

    // 获取输入输出个数
    rknn_input_output_num io_num;
    ret = rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) {
        LOG_ERROR("rknn_query fail! ret=%d", ret);
        return DEVICE_INIT_FAIL;
    }

    input_num_ = io_num.n_input;
    output_num_ = io_num.n_output;

    // 输入属性
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (int i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOG_ERROR("rknn_query fail! ret=%d", ret);
            return INPUT_ATTR_ERROR;
        }
        // set input_shapes_
        input_attrs_.push_back(rknn_tensor_attr_convert(input_attrs[i]));

        is_int8_ = (input_attrs[i].type == RKNN_TENSOR_INT8 ||input_attrs[i].type == RKNN_TENSOR_UINT8) ? 1 : 0;
    }

    // 输出属性
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (int i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret != RKNN_SUCC) {
            LOG_ERROR("rknn_query fail! ret=%d", ret);
            return OUTPUT_ATTR_ERROR;
        }
        // set output_shapes_
        output_attrs_.push_back(rknn_tensor_attr_convert(output_attrs[i]));
    }
    return SUCCESS;
}

error_e RKEngine::destory() {
    if (ctx_created_) {
        rknn_destroy(rknn_ctx_);
    }
    return SUCCESS;
}

void RKEngine::Print() {
    LOG_INFO("****************************************************************************");
    // 获取rknn版本信息
    rknn_sdk_version version;
    rknn_query(rknn_ctx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    LOG_INFO("RKNN API version: %s", version.api_version);
    LOG_INFO("RKNN Driver version: %s", version.drv_version);
    
    LOG_INFO("Infer %p detail", this);
    LOG_INFO("\tInputs: %d", input_num_);
    for(int i = 0; i < input_num_; ++i){
        auto input_attr = input_attrs_[i];
        std::vector<int64_t> shapes_(input_attr.shape.begin(), input_attr.shape.end());
        LOG_INFO(
            "\t\t%d.%s : shape {%s}, %s", 
            i, input_attrs_[i].name.c_str(), 
            iTools::vector_shape_string(shapes_).c_str(), 
            data_type_string(input_attr.type).c_str()
        );
    }

    LOG_INFO("\tOutputs: %d", output_num_);
    for(int i = 0; i < output_num_; ++i){
        auto output_attr = output_attrs_[i];
        std::vector<int64_t> shapes_(output_attr.shape.begin(), output_attr.shape.end());
        LOG_INFO(
            "\t\t%d.%s : shape {%s}, %s", 
            i, output_attrs_[i].name.c_str(), 
            iTools::vector_shape_string(shapes_).c_str(), 
            data_type_string(output_attr.type).c_str()
        );
    }
    LOG_INFO("****************************************************************************");
}

std::vector<int> RKEngine::GetInputShape(int index) {
    auto input_attr = input_attrs_[index];
    return input_attr.shape;
}

std::vector<std::vector<int>> RKEngine::GetOutputShapes() {
    std::vector<std::vector<int>> output;
    for (auto out_attr : output_attrs_) {
        output.push_back(out_attr.shape);
    }
    return output;
}

std::string RKEngine::GetInputType(int index){
    return data_type_string(input_attrs_[index].type);
}

std::vector<std::string> RKEngine::GetOutputTypes(){
    std::vector<std::string> output_chars;
    for (auto output_attr : output_attrs_){
        output_chars.push_back(data_type_string(output_attr.type));
    }
    return output_chars;
}

void RKEngine::BindingInput(InferenceDataType& inputData) {
    if (inputData.size() != input_num_) {
        LOG_ERROR("inputs num not match! inputs.size()=%ld, input_num_=%d", inputData.size(), input_num_);
        exit(1);
    }

    rknn_input rknn_inputs[inputData.size()];
    for (int i = 0; i < inputData.size(); i++) {
        rknn_input input;
        memset(&input, 0, sizeof(input));
        input.index     = input_attrs_[i].index;
        input.type      = input_attrs_[i].type;
        input.size      = input_attrs_[i].size;
        input.fmt       = input_attrs_[i].layout;
        input.buf       = inputData[i].first;
        
        rknn_inputs[i] = input;
    }
    int ret = rknn_inputs_set(rknn_ctx_, (uint32_t)inputData.size(), rknn_inputs);
    if (ret < 0) {
        LOG_ERROR("rknn_inputs_set fail! ret=%d", ret);
        exit(1);
    }
}

void RKEngine::GetInferOutput(InferenceDataType& outputData) {
    int ret = rknn_run(rknn_ctx_, nullptr);
    if (ret < 0) {
        LOG_ERROR("rknn_run fail! ret=%d", ret);
        exit(1);
    }

    rknn_output rknn_outputs[output_num_];
    memset(rknn_outputs, 0, sizeof(rknn_outputs));
    for (int i = 0; i < output_num_; ++i) {
        rknn_outputs[i].want_float = !is_int8_;
    }
    ret = rknn_outputs_get(rknn_ctx_, output_num_, rknn_outputs, NULL);
    if (ret < 0) {
        LOG_ERROR("rknn_outputs_get fail! ret=%d", ret);
    }

    for (int i = 0; i < output_num_; ++i) {
        // TODO: 内存泄漏
        void* outData = malloc(rknn_outputs[i].size);
        memcpy(outData, rknn_outputs[i].buf, rknn_outputs[i].size);
        outputData.emplace_back(std::make_pair(outData, rknn_outputs[i].size));
        free(rknn_outputs[i].buf); // 释放缓存
    }
}

std::shared_ptr<ACEngine> create_engine(const std::string &file_path, bool use_plugins) {
    std::shared_ptr<RKEngine> Instance(new RKEngine());
    if (Instance->create(file_path) != SUCCESS) {
        Instance.reset();
    }
    return Instance;
}