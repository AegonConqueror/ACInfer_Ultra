cmake_minimum_required(VERSION 3.15)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(${TensorRT_INCLUDE_DIR})
include_directories(${OpenCV_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})
include_directories(${ONNXRuntime_INCLUDE_DIR})

file(GLOB_RECURSE common_cpp ${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp)
file(GLOB_RECURSE trt_cpp ${CMAKE_CURRENT_SOURCE_DIR}/TensorRT/*.cpp)
file(GLOB_RECURSE deepsort_cpp ${CMAKE_CURRENT_SOURCE_DIR}/app/deepsort/*.cpp)
# file(GLOB_RECURSE botsort_cpp ${CMAKE_CURRENT_SOURCE_DIR}/app/BotSort/*.cpp)
file(GLOB_RECURSE botsort_cpp ${CMAKE_CURRENT_SOURCE_DIR}/app/BB/*.cpp)

add_library(
    onnx
    ${common_cpp}
    ${CMAKE_CURRENT_SOURCE_DIR}/engine/onnxruntime_engine.cpp
)
target_link_libraries(onnx PUBLIC ${ONNXRuntime_LIBS})

# add_executable(
#     yolov8_track
#     yolov8_track.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/app/detect/yolov8_detector.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/app/yolov8/postprocess.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/app/deepsort.cpp
#     ${deepsort_cpp}
# )
# target_link_libraries(yolov8_track PRIVATE onnx ${OpenCV_LIBS} Eigen3::Eigen)

# add_executable(
#     onnxruntime_inference
#     onnxruntime_inference.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/app/detect/yolov5_detector.cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/app/yolov5/postprocess.cpp
# )
# target_link_libraries(onnxruntime_inference PRIVATE onnx ${OpenCV_LIBS})

# add_executable(
#     convert_video
#     convert_video.cpp
# )
# target_link_libraries(convert_video PRIVATE ${OpenCV_LIBS})

add_executable(
    yolov8_botsort
    yolov8_botsort.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app/detect/yolov8_detector.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app/yolov8/postprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app/ReID.cpp
    ${botsort_cpp}
)
target_link_libraries(yolov8_botsort PRIVATE onnx ${OpenCV_LIBS} Eigen3::Eigen)