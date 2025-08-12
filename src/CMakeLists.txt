cmake_minimum_required(VERSION 3.15)

include_directories(${OpenCV_INCLUDE_DIR})
include_directories(${TensorRT_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIR})

# ============== trt & tools lib ==============
file(GLOB_RECURSE UTILS_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/utils/*.cpp)
file(GLOB_RECURSE TRT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/trt/*.cpp)

add_library(
    trt_utils
    SHARED
    ${UTILS_SOURCES}
    ${TRT_SOURCES}
)
target_link_libraries(trt_utils PUBLIC ${TensorRT_LIBS} ${OpenCV_LIBS} ${CUDA_LIBS})

# ============== engine lib ==============
set(ENGINE_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/engine/trt_engine.cpp")
add_library(
    infer_engine
    SHARED
    ${ENGINE_SOURCES}
)
target_link_libraries(infer_engine PUBLIC trt_utils)

add_executable(
    warpAffine_test
    ${CMAKE_CURRENT_SOURCE_DIR}/demo/warpAffine_test.cpp
)
target_link_libraries(warpAffine_test PRIVATE infer_engine)
set_target_properties(warpAffine_test PROPERTIES CUDA_ARCHITECTURES "61;70;75")