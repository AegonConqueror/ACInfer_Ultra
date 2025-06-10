
#include "yolov8PoseLayerPlugin.h"

#include "trt/trt_cuda.h"

#define NHEADNUM 3
#define NBINPUTS 10

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

nvinfer1::PluginFieldCollection YOLOv8PoseLayerPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> YOLOv8PoseLayerPluginCreator::mPluginAttributes;

YOLOv8PoseLayer::YOLOv8PoseLayer(void const *data, size_t length) {
    const char *d = static_cast<const char *>(data);

    read(d, m_maxStride);
    read(d, m_numClasses);
    read(d, m_keyPoints);
    read(d, m_netWidth);
    read(d, m_netHeight);
    read(d, m_totalAnchors);
    read(d, m_OutputSize);
    read(d, m_socreThreshold);
    read(d, m_nmsThreshold);

    m_mapSize.resize(NHEADNUM);
    for (uint i = 0; i < m_mapSize.size(); i++) {
        read(d, m_mapSize[i]);
    }

    m_headStarts.resize(NHEADNUM);
    for (uint i = 0; i < m_headStarts.size(); i++) {
        read(d, m_headStarts[i]);
    }
}

YOLOv8PoseLayer::YOLOv8PoseLayer(const uint &max_stride, const float &socre_threshold, const float &nms_threshold)
    : m_maxStride(max_stride), m_socreThreshold(socre_threshold), m_nmsThreshold(nms_threshold) {};


nvinfer1::IPluginV2DynamicExt* YOLOv8PoseLayer::clone() const noexcept {
    nvinfer1::IPluginV2DynamicExt* plugin_layer{nullptr};
    try {
        plugin_layer = new YOLOv8PoseLayer(m_maxStride, m_socreThreshold, m_nmsThreshold);
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
    const nvinfer1::IDimensionExpr* batch_size = inputs[0].d[0];

    const nvinfer1::IDimensionExpr* output_num_boxes = exprBuilder.constant(0);

    for (int32_t i = 0; i < NHEADNUM; i++) {
        output_num_boxes = exprBuilder.operation(
            nvinfer1::DimensionOperation::kSUM, 
            *output_num_boxes,
            *inputs[i * 2].d[3]
        );
    }

    if (outputIndex == 0) { // NumDetections [batch_size, 1]
        out_dim.nbDims = 2;
        out_dim.d[0] = batch_size;
        out_dim.d[1] = exprBuilder.constant(1);
    }
    else if (outputIndex == 1) { // DetectionClasses [batch_size, numboxes]
        out_dim.nbDims = 2;
        out_dim.d[0] = batch_size;
        out_dim.d[1] = output_num_boxes;
    }
    else if (outputIndex == 2) { // DetectionScores [batch_size, numboxes]
        out_dim.nbDims = 2;
        out_dim.d[0] = batch_size;
        out_dim.d[1] = output_num_boxes;
    }
    else if (outputIndex == 3) { // DetectionBoxes [batch_size, numboxes, 4]
        out_dim.nbDims = 3;
        out_dim.d[0] = batch_size;
        out_dim.d[1] = output_num_boxes;
        out_dim.d[2] = exprBuilder.constant(4);
    }
    else { // DetectionKeyPoints [batch_size, numboxes, 3]
        out_dim.nbDims = 3;
        out_dim.d[0] = batch_size;
        out_dim.d[1] = output_num_boxes;
        out_dim.d[2] = exprBuilder.constant(3);
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
    size_t totalSize = 0;

    totalSize += sizeof(m_maxStride);
    totalSize += sizeof(m_numClasses);
    totalSize += sizeof(m_keyPoints);
    totalSize += sizeof(m_netWidth);
    totalSize += sizeof(m_netHeight);
    totalSize += sizeof(m_totalAnchors);
    totalSize += sizeof(m_OutputSize);
    totalSize += sizeof(m_socreThreshold);
    totalSize += sizeof(m_nmsThreshold);

    totalSize += m_mapSize.size() * sizeof(m_mapSize[0]);
    totalSize += m_headStarts.size() * sizeof(m_headStarts[0]);

    return totalSize;
}

void YOLOv8PoseLayer::serialize(void* buffer) const noexcept {
    char *d = static_cast<char *>(buffer);

    write(d, m_maxStride);
    write(d, m_numClasses);
    write(d, m_keyPoints);
    write(d, m_netWidth);
    write(d, m_netHeight);
    write(d, m_totalAnchors);
    write(d, m_OutputSize);
    write(d, m_socreThreshold);
    write(d, m_nmsThreshold);

    // write m_mapSize:
    for (int i = 0; i < m_mapSize.size(); i++) {
        write(d, m_mapSize[i]);
    }

    // write m_headStarts:
    for (int i = 0; i < m_headStarts.size(); i++) {
        write(d, m_headStarts[i]);
    }
}

void YOLOv8PoseLayer::configurePlugin(
    nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInput, 
    nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutput
) noexcept {
    assert(nbInput == NBINPUTS);

    m_totalAnchors = 0;
    m_mapSize.clear();
    m_headStarts.clear();
    m_numClasses = 0;
    m_keyPoints = 0;
    for (int i = 0; i < NHEADNUM; i++) {
        m_headStarts.push_back(m_totalAnchors);
        m_totalAnchors += in[i * 2].desc.dims.d[2] * in[i * 2].desc.dims.d[3];
        m_mapSize.push_back(in[i * 2 + 1].desc.dims.d[2]);
    }

    m_netWidth = m_mapSize[0] * m_maxStride;
    m_netHeight = m_mapSize[0] * m_maxStride;

    m_numClasses = in[1].desc.dims.d[1];
    m_keyPoints = static_cast<int>(in[6].desc.dims.d[1] / 3);
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

int32_t YOLOv8PoseLayer::enqueue(
    nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream
) noexcept {
    const int batchSize = inputDesc[0].dims.d[0];

    void* num_detections      = outputs[0];
    void* detection_classes   = outputs[1];
    void* detection_scores    = outputs[2];
    void* detection_boxes     = outputs[3];
    void* detection_keypoints = outputs[4];

    checkCudaRuntime(cudaMemsetAsync((int *)num_detections, 0, sizeof(int) * batchSize, stream));
    checkCudaRuntime(cudaMemsetAsync((int *)detection_classes, 0, sizeof(int) * batchSize * m_totalAnchors, stream));
    checkCudaRuntime(cudaMemsetAsync((float *)detection_scores, 0, sizeof(float) * batchSize * m_totalAnchors, stream));
    checkCudaRuntime(cudaMemsetAsync((float *)detection_boxes, 0, sizeof(float) * batchSize * m_totalAnchors * 4, stream));
    checkCudaRuntime(cudaMemsetAsync((float *)detection_keypoints, 0, sizeof(float) * batchSize * m_totalAnchors * 3, stream));
    
    // TODO: do kernerl
    int* d_mapSize;
    int* d_headStarts;
    checkCudaRuntime(cudaMalloc(&d_mapSize, m_mapSize.size() * sizeof(int)));
    checkCudaRuntime(cudaMalloc(&d_headStarts, m_headStarts.size() * sizeof(int)));
    checkCudaRuntime(cudaMemcpy(d_mapSize, m_mapSize.data(), m_mapSize.size() * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaRuntime(cudaMemcpy(d_headStarts, m_headStarts.data(), m_headStarts.size() * sizeof(int), cudaMemcpyHostToDevice));


    uint64_t inputSize = m_totalAnchors * (4 + m_numClasses + 2 + m_keyPoints * 3);
    
    // for (uint i = 0; i < NBINPUTS; ++i) {
    //     inputDesc[i].dims.nbDims
    // }
    

    return 0;
}

YOLOv8PoseLayerPluginCreator::YOLOv8PoseLayerPluginCreator() {
    mPluginAttributes.emplace_back(nvinfer1::PluginField("max_stride", nullptr, nvinfer1::PluginFieldType::kINT32, NHEADNUM));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("socre_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("nms_threshold", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

nvinfer1::IPluginV2DynamicExt* YOLOv8PoseLayerPluginCreator::createPlugin(
    const char* name, const nvinfer1::PluginFieldCollection* fc
) noexcept {
    const nvinfer1::PluginField *fields = fc->fields;

    int max_stride = 0;
    float socre_threshold = 0.0f;
    float nms_threshold = 0.0f;
    
    for (int i = 0; i < fc->nbFields; i++) {
        const char* attrName = fields[i].name;
        if (!strcmp(attrName, "max_stride")) {
            assert(fields[i].type == nvinfer1::PluginFieldType::kINT32);
            max_stride = *(static_cast<const int *>(fields[i].data));
        }
        if (!strcmp(attrName, "socre_threshold")) {
            assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
            socre_threshold = *(static_cast<const float *>(fields[i].data));
        }
        if (!strcmp(attrName, "nms_threshold")) {
            assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
            nms_threshold = *(static_cast<const float *>(fields[i].data));
        }
    }

    nvinfer1::IPluginV2DynamicExt* plugin_layer{nullptr};
    try {
        plugin_layer = new YOLOv8PoseLayer(max_stride, socre_threshold, nms_threshold);
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
REGISTER_TENSORRT_PLUGIN(YOLOv8PoseLayer);

