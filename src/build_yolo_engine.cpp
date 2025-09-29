
#include <iostream>

#include "tools/ac_utils.h"
#include "builder/trt_builder.h"

int main(int argc, char **argv){

    if (argc != 3) {
        LOG_ERROR("用法: ./bin/build_yolo_engine <onnx_file_source_path> <calibration_data_path>");
        return -1;
    }

    char* onnx_file_path   = argv[1];
    char* data_file_path   = argv[2];

    TRT::QntConfig config = {
        1,
        {0.f, 0.f, 0.f},
        {1.f, 1.f, 1.f},
        TRT::CalibratorType::Entropy,
        data_file_path
    };

    auto engine_builder = TRT::create_builder(onnx_file_path, TRT::Mode::INT8, config);

    engine_builder->compile(onnx_file_path);
    return 0;
}