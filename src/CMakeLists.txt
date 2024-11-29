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

add_library(
    inference_engine
    ${ENGINE_SOURCES}
    ${UTILS_SOURCES}
)
target_link_libraries(inference_engine PUBLIC ${OpenCV_LIBS})

if(USE_ONNXRUNTIME)
    target_link_libraries(inference_engine PUBLIC ${ONNXRuntime_LIBS})
endif()

add_executable(
    yolov8_img_demo
    yolov8_img_demo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app/detect/yolo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app/detect/yolo_postprocess/postprocess.cpp
)
target_link_libraries(yolov8_img_demo PRIVATE inference_engine)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

file(GLOB_RECURSE DEEPSORT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/app/tracker/botsort/*.cpp)

add_executable(
    yolov8_video_demo
    yolov8_video_demo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app/detect/yolo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app/detect/yolo_postprocess/postprocess.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app/feature/ReID.cpp
    ${DEEPSORT_SOURCES}

)
target_link_libraries(yolov8_video_demo PRIVATE inference_engine Eigen3::Eigen)
