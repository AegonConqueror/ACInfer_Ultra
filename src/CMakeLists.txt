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
    target_link_libraries(build_trt_engine PRIVATE -Wl,--no-as-needed customLayerPlugin)
    # target_link_libraries(build_trt_engine PRIVATE -Wl,--no-as-needed customLayerPlugin yolov8PoseLayerPlugin)
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

if(USE_TENSORRT)
    add_executable(
        yolov8_test_trt
        ${CMAKE_CURRENT_SOURCE_DIR}/demo/yolov8_test_trt.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/app/yolo/yolov8.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/app/yolo/process/preprocess.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/app/yolo/process/yolov8_postprocess.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/kernel/yolov8_pose_kernel/yolov8_pose_process.cu
    )
    target_link_libraries(yolov8_test_trt PRIVATE inference_engine)
    target_link_libraries(yolov8_test_trt PRIVATE tensorrt_utils)
    target_link_libraries(yolov8_test_trt PRIVATE -Wl,--no-as-needed customLayerPlugin)
    set_target_properties(yolov8_test_trt PROPERTIES CUDA_ARCHITECTURES "61;70;75")

    add_executable(
        plugin_demo
        ${CMAKE_CURRENT_SOURCE_DIR}/demo/plugin_demo.cpp
    )
    target_link_libraries(plugin_demo PRIVATE inference_engine)
    target_link_libraries(plugin_demo PRIVATE -Wl,--no-as-needed customLayerPlugin)
    set_target_properties(plugin_demo PROPERTIES CUDA_ARCHITECTURES "61;70;75")
endif(USE_TENSORRT)

# if(USE_ONNXRUNTIME)
#     add_executable(
#         yolov8_test_onnx
#         ${CMAKE_CURRENT_SOURCE_DIR}/demo/yolov8_test_onnx.cpp
#         ${CMAKE_CURRENT_SOURCE_DIR}/app/yolo/yolov8.cpp
#         ${CMAKE_CURRENT_SOURCE_DIR}/app/yolo/process/preprocess.cpp
#         ${CMAKE_CURRENT_SOURCE_DIR}/app/yolo/process/yolov8_postprocess.cpp
#     )
#     target_link_libraries(yolov8_test_onnx PRIVATE inference_engine)

#     add_executable(
#         kxwell_pose_demo
#         ${CMAKE_CURRENT_SOURCE_DIR}/demo/kxwell_pose_demo.cpp
#         ${CMAKE_CURRENT_SOURCE_DIR}/app/yolo/yolov8pose.cpp
#     )
#     target_link_libraries(kxwell_pose_demo PRIVATE inference_engine)
# endif()

# find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# file(GLOB_RECURSE DEEPSORT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/app/tracker/botsort/*.cpp)

# add_executable(
#     yolov8_video_demo
#     ${CMAKE_CURRENT_SOURCE_DIR}/demo/yolov8_video_demo.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/app/detect/yolo.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/app/detect/yolo_postprocess/postprocess.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/app/feature/ReID.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/app/tracker/deepsort/deepsort.cpp

# )
# target_link_libraries(yolov8_video_demo PRIVATE inference_engine Eigen3::Eigen)
