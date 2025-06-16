/**
 * *****************************************************************************
 * File name:   yolov8PoseLayerPlugin.h
 * 
 * @brief  自定义的TensorRT yolov8 pose decode plugin
 * 
 * 
 * Created by Aegon on 2025-06-09
 * Copyright © 2025 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACINFER_ULTRA_YOLOV8POSELAYERPLUGIN_H
#define ACINFER_ULTRA_YOLOV8POSELAYERPLUGIN_H

#include <iostream>
#include <memory>
#include <vector>
#include <cassert>
#include <cstring>

#include <cuda_runtime_api.h>
#include "NvInferPlugin.h"

#include "yolov8PoseLayerParameters.h"

namespace {
    const char *YOLOV8POSELAYER_PLUGIN_VERSION{"1"};
    const char *YOLOV8POSELAYER_PLUGIN_NAME{"Yolov8PoseLayer"};
} // namespace

class YOLOv8PoseLayer : public nvinfer1::IPluginV2DynamicExt {
public:
    // ============================= 构造函数与析构函数 =============================
    explicit YOLOv8PoseLayer(YOLOv8PoseLayerParameters params);
    YOLOv8PoseLayer(void const *data, size_t length);
    
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;

    // ============================= 输出相关函数 =============================
    int32_t getNbOutputs() const noexcept override { return 5; }

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t index, nvinfer1::DimsExprs const* inputs, int32_t nbInputDims,
        nvinfer1::IExprBuilder &exprBuilder
    ) noexcept override;

    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs
    ) const noexcept override;

    // ============================= 序列化与反序列化相关函数 =============================
    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    char const* getPluginType() const noexcept override { return YOLOV8POSELAYER_PLUGIN_NAME; }

    char const* getPluginVersion() const noexcept override { return YOLOV8POSELAYER_PLUGIN_VERSION; }

    // ============================= 初始化、配置和销毁函数 =============================
    int32_t initialize() noexcept override { return 0; }

    void terminate() noexcept override {}

    void destroy() noexcept override { delete this; }

    void configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInput, 
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutput
    ) noexcept override;

    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs
    ) noexcept override;

    // ============================= 运行相关函数 =============================
    size_t getWorkspaceSize(
        nvinfer1::PluginTensorDesc const* inputs, int32_t nbInputs, 
        nvinfer1::PluginTensorDesc const* outputs, int32_t nbOutputs
    ) const noexcept override;

    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream
    ) noexcept override;

    // ============================= 需要实现的虚函数函数 =============================
    void setPluginNamespace(nvinfer1::AsciiChar const *plugin_namespace) noexcept override { m_Namespace = plugin_namespace; }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override { return m_Namespace.c_str(); }

private:
    std::string m_Namespace;
    YOLOv8PoseLayerParameters mParam{};
};

class YOLOv8PoseLayerPluginCreator : public nvinfer1::IPluginCreator {
public:
    YOLOv8PoseLayerPluginCreator() noexcept;
    ~YOLOv8PoseLayerPluginCreator() noexcept {}

    const char* getPluginName() const noexcept override { return YOLOV8POSELAYER_PLUGIN_NAME; }
    const char* getPluginVersion() const noexcept override { return YOLOV8POSELAYER_PLUGIN_VERSION; }

    nvinfer1::IPluginV2DynamicExt* createPlugin(
        const char* name, const nvinfer1::PluginFieldCollection* fc
    ) noexcept override;

    nvinfer1::IPluginV2DynamicExt* deserializePlugin(
        const char *name, const void *serialData, size_t serialLength
    ) noexcept override;

    void setPluginNamespace(const char *libNamespace) noexcept override { mNamespace = libNamespace; }

    const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

    const nvinfer1::PluginFieldCollection *getFieldNames() noexcept override { return &mFC; }

private:
    std::string mNamespace;
    YOLOv8PoseLayerParameters mParam;
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};

#endif // ACINFER_ULTRA_YOLOV8POSELAYERPLUGIN_H