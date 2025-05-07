
#include "customLayerPlugin.h"

#include "trt/trt_cuda.h"

nvinfer1::PluginFieldCollection CustomLayer::mFC{};
std::vector<nvinfer1::PluginField> CustomLayer::mPluginAttributes;

cudaError_t custom_layer_inference(float *input, float *offset, float *mul, float *alpha, float *output);

CustomLayerPlugin::CustomLayerPlugin(const nvinfer1::PluginFieldCollection &fc) {

    // 遍历 field_collection 中的每个字段；
    for (int i = 0; i < fc.nbFields; i++) {

        if (!strcmp("alpha", fc.fields[i].name)) {
            // 1. 将其值（一个 float*）转换成浮点数 this->alpha；
            this->alpha = *((float *)fc.fields[i].data);

            // 2. 为 GPU 上的 d_alpha 分配内存；
            checkCudaRuntime(
                cudaMalloc(&this->d_alpha, sizeof(float))
            );

            // 3. 将 alpha 的值从 CPU 复制到 GPU（cudaMemcpy）。
            checkCudaRuntime(
                cudaMemcpy(this->d_alpha, &this->alpha, sizeof(float), cudaMemcpyHostToDevice)
            );
        }
    }
}

CustomLayerPlugin::CustomLayerPlugin(const CustomLayerPlugin &plugin) {
    this->alpha = plugin.alpha;
    checkCudaRuntime(
        cudaMalloc(&this->d_alpha, sizeof(float))
    );
    checkCudaRuntime(
        cudaMemcpy(this->d_alpha, &this->alpha, sizeof(float), cudaMemcpyHostToDevice)
    );
}

CustomLayerPlugin::CustomLayerPlugin(void const *data, size_t length) {
    memcpy(&this->alpha, data, sizeof(this->alpha));
    checkCudaRuntime(
        cudaMalloc(&this->d_alpha, sizeof(float))
    );
    checkCudaRuntime(
        cudaMemcpy(this->d_alpha, &this->alpha, sizeof(float), cudaMemcpyHostToDevice)
    );
}

CustomLayerPlugin::~CustomLayerPlugin() {
    checkCudaRuntime(
        cudaFree(this->d_alpha)
    );
}

nvinfer1::IPluginV2Ext* CustomLayerPlugin::clone() const noexcept {
    nvinfer1::IPluginV2Ext *plugin{nullptr};
    try {
        plugin = new CustomLayerPlugin(*this);
    } catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
    return plugin;
    
}

int32_t CustomLayerPlugin::getNbOutputs() const noexcept { return 1; }

nvinfer1::Dims CustomLayerPlugin::getOutputDimensions(
    int32_t index, nvinfer1::Dims const* inputs, int32_t nbInputDims
) noexcept {
    return inputs[index];
}

nvinfer1::DataType CustomLayerPlugin::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs
) const noexcept {
    return inputTypes[0];
}

size_t CustomLayerPlugin::getSerializationSize() const noexcept {
    size_t volume = 0;
    volume += sizeof(this->alpha);
    return volume;
}

void CustomLayerPlugin::serialize(void* buffer) const noexcept {
    memcpy(buffer, &this->alpha, sizeof(this->alpha));
    return ;
}

char const* CustomLayerPlugin::getPluginType() const noexcept {
    return CUSTOMLAYER_PLUGIN_NAME;
}

char const* CustomLayerPlugin::getPluginVersion() const noexcept {
    return CUSTOMLAYER_PLUGIN_VERSION;
}

int32_t CustomLayerPlugin::initialize() noexcept {
    return 0;
}

void CustomLayerPlugin::terminate() noexcept {
    return ;
}

void CustomLayerPlugin::destroy() noexcept {
    delete this;
}

void CustomLayerPlugin::configurePlugin(
    nvinfer1::PluginTensorDesc const* in, int32_t nbInput, 
    nvinfer1::PluginTensorDesc const* out, int32_t nbOutput
) noexcept {
    return ;
}

bool CustomLayerPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs
) const noexcept {
    return inOut[pos].type == nvinfer1::DataType::kFLOAT;
}

size_t CustomLayerPlugin::getWorkspaceSize(int32_t maxBatchSize) const noexcept {
    return 0;
}

int32_t CustomLayerPlugin::enqueue(
    int32_t batchSize, void const* const* inputs, void* const* outputs, 
    void* workspace, cudaStream_t stream
) noexcept {
    custom_layer_inference(
        (float *)inputs[0],
        (float *)inputs[1],
        (float *)inputs[2],
        this->d_alpha,
        (float *)outputs[0]
    );

    return 0;
}

void CustomLayerPlugin::setPluginNamespace(nvinfer1::AsciiChar const *plugin_namespace) noexcept {
    this->plugin_namespace = plugin_namespace;
}

nvinfer1::AsciiChar const *CustomLayerPlugin::getPluginNamespace() const noexcept {
    return this->plugin_namespace.data();
}

bool CustomLayerPlugin::isOutputBroadcastAcrossBatch(
    int32_t output_idx,
    bool const *is_broadcasted_inputs,
    int32_t num_inputs
) const noexcept {
    return false;
}

bool CustomLayerPlugin::canBroadcastInputAcrossBatch(int32_t input_idx) const noexcept {
    return false;
}

CustomLayer::CustomLayer() {
    // 在创建 PluginCreator 时，就设置好有哪些 field，然后读取模型时，就会将模型中与 field 相同名称的属性一个一个取出来
    mPluginAttributes.emplace_back(
        nvinfer1::PluginField("alpha", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1)
    );

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

CustomLayer::~CustomLayer() {

}

char const* CustomLayer::getPluginName() const noexcept {
    return CUSTOMLAYER_PLUGIN_NAME;
}

char const* CustomLayer::getPluginVersion() const noexcept {
    return CUSTOMLAYER_PLUGIN_VERSION;
}

nvinfer1::PluginFieldCollection const* CustomLayer::getFieldNames() noexcept {
    return &mFC;
}

nvinfer1::IPluginV2Ext* CustomLayer::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc
) noexcept {
    nvinfer1::IPluginV2Ext *plugin{nullptr};
    if (fc != nullptr) {
        try {
            plugin = new CustomLayerPlugin(*fc);
            mFC = *fc;
        }
        catch(const std::exception& e) {
            std::cerr << e.what() << '\n';
        }
    }
    return plugin;
}

nvinfer1::IPluginV2Ext* CustomLayer::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength
) noexcept {
    nvinfer1::IPluginV2Ext *plugin{nullptr};
    try {
        plugin = new CustomLayerPlugin(serialData, serialLength);
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
    return plugin;
}

void CustomLayer::setPluginNamespace(nvinfer1::AsciiChar const *plugin_namespace) noexcept {
    this->mNamespace = plugin_namespace;
}

nvinfer1::AsciiChar const* CustomLayer::getPluginNamespace() const noexcept {
    return this->mNamespace.c_str();
}

// 注册插件。 在实现了各个类方法后，需要调用宏对plugin进行注册。以方便TensorRT识别并找到对应的Plugin。
REGISTER_TENSORRT_PLUGIN(CustomLayer);