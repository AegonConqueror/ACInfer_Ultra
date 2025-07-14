
#include "yolov8PoseLayerPlugin.h"

#include "trt/trt_cuda.h"

namespace {
    template <typename T>
    void write(char *&buffer, const T &val) {
        *reinterpret_cast<T *>(buffer) = val;
        buffer += sizeof(T);
    }

    template <typename T>
    void read(const char *&buffer, T &val) {
        val = *reinterpret_cast<const T *>(buffer);
        buffer += sizeof(T);
    }
}

pluginStatus_t YOLOv8PoseLayerInference(
    YOLOv8PoseLayerParameters param,
    const void* regInput, const void* clsInput, const void* psInput,
    void* numDetectionsOutput, void* nmsClassesOutput, void* nmsScoresOutput, 
    void* nmsBoxesOutput, void* nmsKeyPointsOutput, void* workspace, cudaStream_t stream
);

nvinfer1::PluginFieldCollection YOLOv8PoseLayerPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> YOLOv8PoseLayerPluginCreator::mPluginAttributes;

YOLOv8PoseLayer::YOLOv8PoseLayer(void const *data, size_t length) {
    const char *d = static_cast<const char *>(data);
    read(d, mParam);
}

YOLOv8PoseLayer::YOLOv8PoseLayer(YOLOv8PoseLayerParameters param): mParam(param) {};

nvinfer1::IPluginV2DynamicExt* YOLOv8PoseLayer::clone() const noexcept {
    nvinfer1::IPluginV2DynamicExt* plugin_layer{nullptr};
    try {
        plugin_layer = new YOLOv8PoseLayer(mParam);
        plugin_layer->setPluginNamespace(m_Namespace.c_str());
        return plugin_layer;
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
    return nullptr;
}

nvinfer1::DimsExprs YOLOv8PoseLayer::getOutputDimensions(
    int32_t outputIndex, nvinfer1::DimsExprs const* inputs, int32_t nbInputDims,
    nvinfer1::IExprBuilder &exprBuilder
) noexcept {
    assert(outputIndex < 5);

    nvinfer1::DimsExprs out_dim;

    if (outputIndex == 0) { // NumDetections [batch_size, 1]
        out_dim.nbDims = 2;
        out_dim.d[0] = inputs[0].d[0];
        out_dim.d[1] = exprBuilder.constant(1);
    }
    else if (outputIndex == 1) { // DetectionClasses [batch_size, numboxes]
        out_dim.nbDims = 2;
        out_dim.d[0] = inputs[0].d[0];
        out_dim.d[1] = exprBuilder.constant(mParam.numOutputBoxes);
    }
    else if (outputIndex == 2) { // DetectionScores [batch_size, numboxes]
        out_dim.nbDims = 2;
        out_dim.d[0] = inputs[0].d[0];
        out_dim.d[1] = exprBuilder.constant(mParam.numOutputBoxes);
    }
    else if (outputIndex == 3) { // DetectionBoxes [batch_size, numboxes, 4]
        out_dim.nbDims = 3;
        out_dim.d[0] = inputs[0].d[0];
        out_dim.d[1] = exprBuilder.constant(mParam.numOutputBoxes);
        out_dim.d[2] = exprBuilder.constant(4);
    }
    else { // DetectionKeyPoints [batch_size, numboxes, 3]
        out_dim.nbDims = 3;
        out_dim.d[0] = inputs[0].d[0];
        out_dim.d[1] = exprBuilder.constant(mParam.numOutputBoxes);
        out_dim.d[2] = exprBuilder.constant(3 * mParam.numKeypoints);
    }
    return out_dim;
}

nvinfer1::DataType YOLOv8PoseLayer::getOutputDataType(
    int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs
) const noexcept {
    // num_detection and classes
    if (index == 0 || index == 1) {
        return nvinfer1::DataType::kINT32;
    }
    return inputTypes[0];
}

size_t YOLOv8PoseLayer::getSerializationSize() const noexcept {
    return sizeof(YOLOv8PoseLayerParameters);
}

void YOLOv8PoseLayer::serialize(void* buffer) const noexcept {
    char *d = static_cast<char *>(buffer);
    write(d, mParam);
}

void YOLOv8PoseLayer::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInput, 
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutput
) noexcept {
    assert(nbInput == 3);

    mParam.numClasses = in[1].desc.dims.d[1];

    mParam.numAnchors = in[0].desc.dims.d[3];
}

bool YOLOv8PoseLayer::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs
) noexcept {
    if (inOut[pos].format != nvinfer1::PluginFormat::kLINEAR) {
        return false;
    }

    const int posOut = pos - nbInputs;

    if (posOut == 0 || posOut == 1) {
        return inOut[pos].type == nvinfer1::DataType::kINT32 && inOut[pos].format == nvinfer1::PluginFormat::kLINEAR;
    }

    // all other inputs/outputs: fp32 or fp16
    return (inOut[pos].type == nvinfer1::DataType::kFLOAT) && (inOut[0].type == inOut[pos].type);
}

size_t YOLOv8PoseLayerWorkspaceSize(int batchSize, int numAnchors) {

    size_t size_rects =  batchSize * numAnchors * (4 + 1) * sizeof(float);
    size_t size_classes =  batchSize * numAnchors * sizeof(int);
    size_t size_object_count = batchSize * sizeof(int);
    size_t size_keep =  batchSize * numAnchors * sizeof(int);

    return size_rects + size_classes + size_object_count + size_keep;
}

size_t YOLOv8PoseLayer::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs, 
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs
) const noexcept {
    int batchSize = inputs[0].dims.d[0];
    int numAnchors = inputs[0].dims.d[3];
    
    return YOLOv8PoseLayerWorkspaceSize(batchSize, numAnchors);
}

int32_t YOLOv8PoseLayer::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream
) noexcept { 
    try {
        mParam.batchSize = inputDesc[0].dims.d[0];

        const void* const regInput = inputs[0];
        const void* const clsInput = inputs[1];
        const void* const psInput  = inputs[2];

        void* numDetectionsOutput = outputs[0];
        void* nmsClassesOutput    = outputs[1];
        void* nmsScoresOutput     = outputs[2];
        void* nmsBoxesOutput      = outputs[3];
        void* nmsKeyPointsOutput  = outputs[4];

        return YOLOv8PoseLayerInference(
            mParam,
            regInput, clsInput, psInput,
            numDetectionsOutput, nmsClassesOutput, nmsScoresOutput, 
            nmsBoxesOutput, nmsKeyPointsOutput, workspace, stream
        );
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
    return -1;
}

YOLOv8PoseLayerPluginCreator::YOLOv8PoseLayerPluginCreator() noexcept {
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField("input_width", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("input_height", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("max_output_boxes", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("min_stride", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("num_keypoints", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("socre_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("nms_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

nvinfer1::IPluginV2DynamicExt* YOLOv8PoseLayerPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc
) noexcept {

    nvinfer1::IPluginV2DynamicExt* plugin_layer{nullptr};

    try {
        const nvinfer1::PluginField* fields = fc->fields;

        for (int i = 0; i < fc->nbFields; i++) {
            const char* attrName = fields[i].name;
            if (!strcmp(attrName, "input_width")) {
                assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                mParam.inputWidth= *(static_cast<const int *>(fields[i].data));
            }
            if (!strcmp(attrName, "input_height")) {
                assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                mParam.inputHeight= *(static_cast<const int *>(fields[i].data));
            }
            if (!strcmp(attrName, "max_output_boxes")) {
                assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                mParam.numOutputBoxes = *(static_cast<const int *>(fields[i].data));
            }
            if (!strcmp(attrName, "min_stride")) {
                assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                mParam.minStride = *(static_cast<const int *>(fields[i].data));
            }
            if (!strcmp(attrName, "num_keypoints")) {
                assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                mParam.numKeypoints = *(static_cast<const int *>(fields[i].data));
            }
            if (!strcmp(attrName, "socre_threshold")) {
                assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
                mParam.scoreThreshold = *(static_cast<const float *>(fields[i].data));
            }
            if (!strcmp(attrName, "nms_threshold")) {
                assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
                mParam.iouThreshold = *(static_cast<const float *>(fields[i].data));
            }
        }

        plugin_layer = new YOLOv8PoseLayer(mParam);
        plugin_layer->setPluginNamespace(mNamespace.c_str());
        return plugin_layer;
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
    }

    return nullptr;
}

nvinfer1::IPluginV2DynamicExt* YOLOv8PoseLayerPluginCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength
) noexcept {
    nvinfer1::IPluginV2DynamicExt *plugin{nullptr};
    try {
        plugin = new YOLOv8PoseLayer(serialData, serialLength);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
    return nullptr;
}
// 注册插件。 在实现了各个类方法后，需要调用宏对plugin进行注册。以方便TensorRT识别并找到对应的Plugin。
REGISTER_TENSORRT_PLUGIN(YOLOv8PoseLayerPluginCreator);

