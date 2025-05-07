/**
 * *****************************************************************************
 * File name:   customLayerPlugin.h
 * 
 * @brief  第一个自定义的TensorRT static shape plugin
 * 
 * 
 * Created by Aegon on 2025-05-07
 * Copyright © 2025 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACINFER_ULTRA_CUSTOMLAYERPLUGIN_H
#define ACINFER_ULTRA_CUSTOMLAYERPLUGIN_H

#include <iostream>
#include <memory>
#include <vector>
#include <cassert>
#include <cstring>

#include <cuda_runtime_api.h>
#include "NvInferPlugin.h"

/**
 * @brief  定义插件版本和插件名
 * 
 */
namespace {
    const char *CUSTOMLAYER_PLUGIN_VERSION{"1"};
    const char *CUSTOMLAYER_PLUGIN_NAME{"CustomLayer"};
} // namespace

/**
 * @brief  实现插件类。
 * 
 * 插件类需要继承IPluginV2DynamicExt类，并实现其中一些成员变量和函数
 * 
 */
class CustomLayerPlugin : public nvinfer1::IPluginV2IOExt {
public:

    // ============================= 构造函数与析构函数 =============================
    /**
     * @brief  该构造函数用于从模型构建过程中的插件参数集合中提取参数值
     * 
     */
    CustomLayerPlugin(const nvinfer1::PluginFieldCollection &fc);

    /**
     * @brief  拷贝构造函数
     * 
     */
    CustomLayerPlugin(const CustomLayerPlugin &plugin);

    /**
     * @brief  从序列化数据反序列化构建
     * 
     */
    CustomLayerPlugin(void const *data, size_t length);

    CustomLayerPlugin() = delete;

    ~CustomLayerPlugin() override;

    nvinfer1::IPluginV2Ext* clone() const noexcept override;

    // ============================= 输出相关函数 =============================
    int32_t getNbOutputs() const noexcept override;

    nvinfer1::Dims getOutputDimensions(
        int32_t index, nvinfer1::Dims const* inputs, int32_t nbInputDims
    ) noexcept override;

    nvinfer1::DataType getOutputDataType(
        int32_t index, nvinfer1::DataType const* inputTypes, int32_t nbInputs
    ) const noexcept override;

    // ============================= 序列化与反序列化相关函数 =============================
    size_t getSerializationSize() const noexcept override;

    void serialize(void* buffer) const noexcept override;

    char const* getPluginType() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    // ============================= 初始化、配置和销毁函数 =============================
    int32_t initialize() noexcept override;

    void terminate() noexcept override;

    void destroy() noexcept override;

    void configurePlugin(
        nvinfer1::PluginTensorDesc const* in, int32_t nbInput, 
        nvinfer1::PluginTensorDesc const* out, int32_t nbOutput
    ) noexcept override;

    bool supportsFormatCombination(
        int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs
    ) const noexcept override;

    // ============================= 运行相关函数 =============================
    size_t getWorkspaceSize(int32_t maxBatchSize) const noexcept override;

    int32_t enqueue(
        int32_t batchSize, void const* const* inputs, void* const* outputs, 
        void* workspace, cudaStream_t stream
    ) noexcept override;

    // ============================= 需要实现的虚函数函数 =============================
    void setPluginNamespace(nvinfer1::AsciiChar const *plugin_namespace) noexcept override;

    nvinfer1::AsciiChar const *getPluginNamespace() const noexcept override;

    bool isOutputBroadcastAcrossBatch(
        int32_t output_idx,
        bool const *is_broadcasted_inputs,
        int32_t num_inputs
    ) const noexcept override;

    bool canBroadcastInputAcrossBatch(int32_t input_idx) const noexcept override;

private:
    std::string plugin_namespace{""};
    float alpha;
    float *d_alpha;
};

/**
 * @brief  实现插件创建类。
 * 
 * 插件创建类需要继承IPluginCreator类，并实现其中一些成员变量和函数。
 * 
 */
class CustomLayer : public nvinfer1::IPluginCreator {
public:
    CustomLayer();
    ~CustomLayer() override;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;

    nvinfer1::IPluginV2Ext* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc
    ) noexcept override;

    nvinfer1::IPluginV2Ext* deserializePlugin(
        char const* name, void const* serialData, size_t serialLength
    ) noexcept override;

    void setPluginNamespace(nvinfer1::AsciiChar const *plugin_namespace) noexcept override;

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override;

private:
    std::string mNamespace{""};
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
};


#endif // ACINFER_ULTRA_CUSTOMLAYERPLUGIN_H