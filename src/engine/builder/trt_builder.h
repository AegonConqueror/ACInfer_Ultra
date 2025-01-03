
#ifndef ACINFER_ULTRA_TRT_BUILDER_H
#define ACINFER_ULTRA_TRT_BUILDER_H

#include <vector>
#include <memory>
#include <functional>

#include "trt/trt_tensor.h"

namespace TRT {
    
    typedef std::function<void(int current, int count, const std::vector<std::string>& files, std::shared_ptr<Tensor>& tensor)> Int8Process;

    enum class Mode : int{
        FP32 = 0,
		FP16 = 1,
		INT8 = 2
    };

    enum class CalibratorType : int{
        MinMax,
        Entropy
    };

    class Model {
    public:
        Model() = default;
        Model(const std::string& onnx_model_path);
        Model(const char* onnx_model_path);

        std::string onnxmodel()         const;
        std::string descript()          const;

        static Model onnx(const std::string &file);
    private:
        std::string onnx_model_path_;
    };

    bool compile(
		Mode mode,
		unsigned int maxBatchSize,
		const Model& onnxSource,
		const std::string& saveto,
		Int8Process int8process = nullptr,
		const std::string& int8ImageDirectory = "",
		const std::string& int8CalibratorCacheFile = "",
        CalibratorType calibratorType = CalibratorType::Entropy,
		const size_t maxWorkspaceSize = 1ul << 30 // 1GB
	);

} // namespace TRT


#endif // ACINFER_ULTRA_TRT_BUILDER_H