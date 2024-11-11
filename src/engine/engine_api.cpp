
#include "engine_api.h"

MS_DLL error_e  Initialize(
    AC_HANDLE* handle, 
    int platform,
    const std::string &file_path,
    bool model_log,
    bool use_plugins
) {
    ACEngine* engine_;
    if (platform == 0) {
#ifdef USE_ONNXRUNTIME
        engine_ = new ONNXEngine();
#endif
    } else {
        LOG_ERROR("Unsupport platform %d", platform);
    }

    error_e ret = engine_->Initialize(file_path, use_plugins);
    *handle = (void*)engine_;

    if (model_log) {
        engine_->Print();
    }
    
    return ret;
}

MS_DLL error_e Destory(AC_HANDLE handle) {
    ACEngine* engine_ = (ACEngine*) handle;
    error_e ret = engine_->Destory();
    delete engine_;
    engine_ = nullptr;
    return ret;
}

MS_DLL void BindingInput(AC_HANDLE handle, InferenceDataType& inputData) {
    ACEngine* engine_ = (ACEngine*) handle;
    engine_->BindingInput(inputData);
}
 
MS_DLL void GetInferOutput(AC_HANDLE handle, InferenceDataType& outputData) {
    ACEngine* engine_ = (ACEngine*) handle;
    engine_->GetInferOutput(outputData);
}

MS_DLL std::vector<int> GetInputShape(AC_HANDLE handle, int index) {
    ACEngine* engine_ = (ACEngine*) handle;
    return engine_->GetInputShape(index);
}

MS_DLL std::string GetInputType(AC_HANDLE handle, int index) {
    ACEngine* engine_ = (ACEngine*) handle;
    return engine_->GetInputType(index);
}

MS_DLL std::vector<std::vector<int>> GetOutputShapes(AC_HANDLE handle) {
    ACEngine* engine_ = (ACEngine*) handle;
    return engine_->GetOutputShapes();
}