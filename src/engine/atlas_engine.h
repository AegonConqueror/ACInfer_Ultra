/**
 * *****************************************************************************
 * File name:   atlas_engine.h
 * 
 * @brief  Atlas inference engine
 * 
 * 
 * Created by Aegon on 2023-07-02
 * Copyright © 2023 House Targaryen. All rights reserved.
 * *****************************************************************************
 */
#ifndef ACINFER_ULTRA_ATLAS_ENGINE_H
#define ACINFER_ULTRA_ATLAS_ENGINE_H

#include "engine.h"
#include "acl/acl.h"

class AtlasEngine : public ACEngine {
public:
    AtlasEngine();
    ~AtlasEngine() override;

    virtual error_e     Initialize(const std::string &file_path, bool owner_device, bool use_plugins=false) override;
    virtual error_e     Destory() override;
    virtual void        BindingInput(InferenceDataType& inputData) override;
    virtual void        GetInferOutput(InferenceDataType& outputData) override;

    virtual void Print() override;

    virtual std::vector<int>                GetInputShape(int index) override;
    virtual std::vector<std::vector<int>>   GetOutputShapes() override;
    virtual std::string                     GetInputType(int index) override;
    virtual std::vector<std::string>        GetOutputTypes() override;

private:
    void UnloadModel();
    void DestroyInput();
    void DestroyOutput();
    void DestroyResource();

private:
    int                             deviceId_;
    bool                            isDevice_;
    bool                            owner_device_;
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
};

#endif // ACINFER_ULTRA_ATLAS_ENGINE_H