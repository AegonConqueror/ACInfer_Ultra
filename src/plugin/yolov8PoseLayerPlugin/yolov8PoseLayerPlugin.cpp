
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
    const void* reg1Input, const void* reg2Input, const void* reg3Input,
    const void* cls1Input, const void* cls2Input, const void* cls3Input,
    const void* ps1Input, const void* ps2Input, const void* ps3Input,
    void* numDetectionsOutput, void* nmsClassesOutput, void* nmsScoresOutput, 
    void* nmsBoxesOutput, void* nmsKeyPointsOutput, void* workspace, cudaStream_t stream
);

nvinfer1::PluginFieldCollection YOLOv8PoseLayerPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> YOLOv8PoseLayerPluginCreator::mPluginAttributes;

YOLOv8PoseLayer::YOLOv8PoseLayer(void const *data, size_t length) {
    const char *d = static_cast<const char *>(data);
    read(d, mParam);
}

YOLOv8PoseLayer::YOLOv8PoseLayer(YOLOv8PoseLayerParameters params) {};

nvinfer1::IPluginV2DynamicExt* YOLOv8PoseLayer::clone() const noexcept {
    nvinfer1::IPluginV2DynamicExt* plugin_layer{nullptr};
    try {
        plugin_layer = new YOLOv8PoseLayer(mParam);
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
    return plugin_layer;
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
    assert(nbInput == 9);

    mParam.numClasses = in[1].desc.dims.d[1];
    mParam.numKeypoints = static_cast<int>(in[6].desc.dims.d[1] / 3);
    mParam.inputWidth = in[1].desc.dims.d[3] * mParam.minStride;
    mParam.inputHeight = in[1].desc.dims.d[2] * mParam.minStride;

    for (size_t i = 0; i < 3; i++) {
        assert(in[i * 2].desc.dims.nbDims == 4);
        mParam.numAnchors += in[i * 2].desc.dims.d[3];
        if (i == 0)
            mParam.headStart = mParam.numAnchors;
        else if (i == 1)
            mParam.headEnd = mParam.numAnchors;
    }

    for (size_t i = 0; i < 4; i++) {
        mParam.reg1Size = in[0].desc.dims.d[1] * in[0].desc.dims.d[2] * in[0].desc.dims.d[3];
        mParam.reg2Size = in[2].desc.dims.d[1] * in[2].desc.dims.d[2] * in[2].desc.dims.d[3];
        mParam.reg3Size = in[4].desc.dims.d[1] * in[4].desc.dims.d[2] * in[4].desc.dims.d[3];

        mParam.cls1Size = in[1].desc.dims.d[1] * in[1].desc.dims.d[2] * in[1].desc.dims.d[3];
        mParam.cls2Size = in[3].desc.dims.d[1] * in[3].desc.dims.d[2] * in[3].desc.dims.d[3];
        mParam.cls3Size = in[5].desc.dims.d[1] * in[5].desc.dims.d[2] * in[5].desc.dims.d[3];

        mParam.ps1Size = in[6].desc.dims.d[1] * in[6].desc.dims.d[2] * in[6].desc.dims.d[3];
        mParam.ps2Size = in[7].desc.dims.d[1] * in[7].desc.dims.d[2] * in[7].desc.dims.d[3];
        mParam.ps3Size = in[8].desc.dims.d[1] * in[8].desc.dims.d[2] * in[8].desc.dims.d[3];
    }
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

    size_t size_rects = (4 + 1) * sizeof(int) * batchSize * numAnchors;
    size_t size_socres = sizeof(float) * batchSize * numAnchors;
    size_t size_object_count = sizeof(int) * batchSize;
    size_t size_keep = sizeof(int) * batchSize * numAnchors;

    return size_rects + size_socres + size_object_count + size_keep;
}

size_t YOLOv8PoseLayer::getWorkspaceSize(
    nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs, 
    nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs
) const noexcept {
    int batchSize = inputs[1].dims.d[0];
    int stride_1_anchors = inputs[0].dims.d[3];
    int stride_2_anchors = inputs[2].dims.d[3];
    int stride_3_anchors = inputs[4].dims.d[3];
    int numAnchors = stride_1_anchors + stride_2_anchors + stride_3_anchors;
    return YOLOv8PoseLayerWorkspaceSize(batchSize, numAnchors);
}

int32_t YOLOv8PoseLayer::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream
) noexcept { 
    try {
        mParam.batchSize = inputDesc[0].dims.d[0];

        const void* const reg1Input = inputs[0];
        const void* const cls1Input = inputs[1];
        const void* const reg2Input = inputs[2];
        const void* const cls2Input = inputs[3];
        const void* const reg3Input = inputs[4];
        const void* const cls3Input = inputs[5];
        const void* const ps1Input  = inputs[6];
        const void* const ps2Input  = inputs[7];
        const void* const ps3Input  = inputs[8];

        void* numDetectionsOutput = outputs[0];
        void* nmsClassesOutput    = outputs[1];
        void* nmsScoresOutput     = outputs[2];
        void* nmsBoxesOutput      = outputs[3];
        void* nmsKeyPointsOutput  = outputs[4];
        
        return YOLOv8PoseLayerInference(
            mParam,
            reg1Input, reg2Input, reg3Input,
            cls1Input, cls2Input, cls3Input,
            ps1Input, ps2Input, ps3Input,
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
    mPluginAttributes.emplace_back(nvinfer1::PluginField("max_output_boxes", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("min_stride", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
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
            if (!strcmp(attrName, "max_output_boxes")) {
                assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                mParam.numOutputBoxes = *(static_cast<const int *>(fields[i].data));
            }
            if (!strcmp(attrName, "min_stride")) {
                assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
                mParam.minStride = *(static_cast<const int *>(fields[i].data));
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
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
    }

    return plugin_layer;
}

nvinfer1::IPluginV2DynamicExt* YOLOv8PoseLayerPluginCreator::deserializePlugin(
    const char *name, const void *serialData, size_t serialLength
) noexcept {
    nvinfer1::IPluginV2DynamicExt *plugin{nullptr};
    try {
        plugin = new YOLOv8PoseLayer(serialData, serialLength);
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
    return plugin;
}
// 注册插件。 在实现了各个类方法后，需要调用宏对plugin进行注册。以方便TensorRT识别并找到对应的Plugin。
REGISTER_TENSORRT_PLUGIN(YOLOv8PoseLayerPluginCreator);

