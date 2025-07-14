cmake_minimum_required(VERSION 3.15)

file(GLOB_RECURSE UTILS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp)

include_directories(${OpenCV_INCLUDE_DIR})

if(USE_ONNXRUNTIME)
    include_directories(${ONNXRuntime_INCLUDE_DIR})
    set(ENGINE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/engine/onnx_engine.cpp")
endif(USE_ONNXRUNTIME)

if(USE_ATALS)
    set(ENGINE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/engine/atlas_engine.cpp")
endif(USE_ATALS)

if(USE_TENSORRT)
    add_subdirectory(plugin)

    include_directories(${TensorRT_INCLUDE_DIR})
    include_directories(${CUDA_INCLUDE_DIR})

    file(GLOB_RECURSE TRT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/trt/*.cpp)
    set(ENGINE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/engine/trt_engine.cpp")

    add_library(
        tensorrt_utils
        SHARED
        ${TRT_SOURCES}
        ${UTILS_SOURCES}
    )
    target_link_libraries(tensorrt_utils PUBLIC ${TensorRT_LIBS} ${OpenCV_LIBS} ${CUDA_LIBS})

    add_executable(
        build_trt_engine
        ${CMAKE_CURRENT_SOURCE_DIR}/demo/build_trt_engine.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/engine/builder/trt_builder.cpp
    )
    target_link_libraries(build_trt_engine PRIVATE tensorrt_utils)
    target_link_libraries(build_trt_engine PRIVATE -Wl,--no-as-needed yolov8PoseLayerPlugin)
    set_target_properties(build_trt_engine PROPERTIES CUDA_ARCHITECTURES "61;70;75")

endif(USE_TENSORRT)

add_library(
    inference_engine
    SHARED
    ${ENGINE_SOURCES}
    ${UTILS_SOURCES}
)
target_link_libraries(inference_engine PUBLIC ${OpenCV_LIBS})

if(USE_ONNXRUNTIME)
    target_link_libraries(inference_engine PUBLIC ${ONNXRuntime_LIBS})
endif()

if(USE_TENSORRT)
    target_link_libraries(inference_engine PUBLIC tensorrt_utils)
endif()

if(USE_ONNXRUNTIME)
    add_library(
        yolo_inference
        SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/app/yolo/yolo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/app/yolo/process/preprocess.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/app/yolo/process/postprocess.cpp
    )
    target_link_libraries(yolo_inference PUBLIC inference_engine)

    add_executable(
        yolo_test_onnx
        ${CMAKE_CURRENT_SOURCE_DIR}/demo/yolo_test_onnx.cpp
    )
    target_link_libraries(yolo_test_onnx PRIVATE yolo_inference)
endif()


if(USE_TENSORRT)
    add_library(
        yolo_inference
        SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/app/yolo/yolo.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/app/yolo/process/preprocess.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/app/yolo/process/postprocess.cpp
    )
    target_link_libraries(yolo_inference PUBLIC inference_engine)


    add_executable(
        yolo_test_trt
        ${CMAKE_CURRENT_SOURCE_DIR}/demo/yolo_test_trt.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel/yolov8_pose_kernel/yolov8_pose_process.cu
    )
    target_link_libraries(yolo_test_trt PRIVATE yolo_inference)
    target_link_libraries(yolo_test_trt PRIVATE -Wl,--no-as-needed yolov8PoseLayerPlugin)
    set_target_properties(yolo_test_trt PROPERTIES CUDA_ARCHITECTURES "61;70;75")
endif(USE_TENSORRT)
