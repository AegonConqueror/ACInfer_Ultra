cmake_minimum_required(VERSION 3.15)

file(GLOB_RECURSE UTILS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp)

cmake_minimum_required(VERSION 3.15)

if(USE_ONNXRUNTIME)
    set(ENGINE_HEADS  
        "${CMAKE_CURRENT_SOURCE_DIR}/engine/engine.h" 
        "${CMAKE_CURRENT_SOURCE_DIR}/engine/onnx_engine.h" 
        "${CMAKE_CURRENT_SOURCE_DIR}/engine/engine_api.h")   
    set(ENGINE_SOURCES  
        "${CMAKE_CURRENT_SOURCE_DIR}/engine/onnx_engine.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/engine/engine_api.cpp")
endif()

include_directories(${OpenCV_INCLUDE_DIR})

if(USE_ONNXRUNTIME)
    include_directories(${ONNXRuntime_INCLUDE_DIR})
endif()

add_executable(
    yolov8_img_demo
    yolov8_img_demo.cpp
    ${ENGINE_SOURCES}
    ${UTILS_SOURCES}
    ${CMAKE_CURRENT_SOURCE_DIR}/app/detect/yolo.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/app/detect/yolo_postprocess/postprocess.cpp
)

target_link_libraries(yolov8_img_demo PUBLIC ${OpenCV_LIBS})

if(USE_ONNXRUNTIME)
    target_link_libraries(yolov8_img_demo PUBLIC ${ONNXRuntime_LIBS})
endif()
