#ifndef ACINFER_ULTRA_TRT_BUILDER_H
#define ACINFER_ULTRA_TRT_BUILDER_H

#include <vector>
#include <memory>
#include <string>
#include <functional>

namespace TRT {

    enum class Mode : int{
        FP32 = 0,
		FP16 = 1,
		INT8 = 2
    };

    enum class CalibratorType : int {
        Entropy     = 0,
        MinMax      = 1,
        None        = 2
    } ;

    struct QntConfig {
        int             max_batch;
        float           mean[3];
        float           std[3];
        CalibratorType  qnt_type = CalibratorType::None;
        std::string     img_Dir;
		// std::string     calibrator_cache;
    };

    class ACBuilder {
    public:
        virtual bool compile(const std::string& onnx_path) = 0;
    };

    std::shared_ptr<ACBuilder> create_builder(const std::string& onnx_path, const Mode mode, const QntConfig& qnt_cfg);
    
} // namespace TRT


#endif // ACINFER_ULTRA_TRT_BUILDER_H