include(FindPackageHandleStandardArgs)

set(TensorRT_SEARCH_PATH 
    PATHS 
    "/home/aegon/envs/TensorRT-10.9.0.34/include"
    "/home/aegon/envs/TensorRT-10.9.0.34/samples"
    "/home/aegon/envs/TensorRT-10.9.0.34/lib"
)

set(TensorRT_ALL_LIBS
    nvinfer
    nvinfer_plugin
    nvonnxparser
)

set(TensorRT_LIBS_LIST)
set(TensorRT_LIBS)

find_path(
    TensorRT_INCLUDE_DIRS
    NAMES NvInfer.h
    PATHS ${TensorRT_SEARCH_PATH}
)

if(TensorRT_INCLUDE_DIRS AND EXISTS "${TensorRT_INCLUDE_DIRS}/NvInferVersion.h")
  file(STRINGS "${TensorRT_INCLUDE_DIRS}/NvInferVersion.h" TensorRT_MAJOR REGEX "^#define NV_TENSORRT_MAJOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIRS}/NvInferVersion.h" TensorRT_MINOR REGEX "^#define NV_TENSORRT_MINOR [0-9]+.*$")
  file(STRINGS "${TensorRT_INCLUDE_DIRS}/NvInferVersion.h" TensorRT_PATCH REGEX "^#define NV_TENSORRT_PATCH [0-9]+.*$")

  string(REGEX REPLACE "^#define NV_TENSORRT_MAJOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MAJOR "${TensorRT_MAJOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_MINOR ([0-9]+).*$" "\\1" TensorRT_VERSION_MINOR "${TensorRT_MINOR}")
  string(REGEX REPLACE "^#define NV_TENSORRT_PATCH ([0-9]+).*$" "\\1" TensorRT_VERSION_PATCH "${TensorRT_PATCH}")
  set(TensorRT_VERSION_STRING "${TensorRT_VERSION_MAJOR}.${TensorRT_VERSION_MINOR}.${TensorRT_VERSION_PATCH}")
endif()

foreach(lib ${TensorRT_ALL_LIBS})
    find_library(
      TensorRT_${lib}_LIBRARY
      NAMES ${lib} 
      PATHS ${TensorRT_SEARCH_PATH}
    )
    set(TensorRT_LIBS_VARS TensorRT_${lib}_LIBRARY ${TensorRT_LIBS_LIST})
    list(APPEND TensorRT_LIBS_LIST TensorRT_${lib}_LIBRARY)
endforeach()

find_package_handle_standard_args(TensorRT REQUIRED_VARS TensorRT_INCLUDE_DIRS ${TensorRT_LIBS_VARS})

if(TensorRT_FOUND)
  foreach(lib ${TensorRT_LIBS_LIST} )
    list(APPEND TensorRT_LIBS ${${lib}})
  endforeach()
  message(STATUS ">>\t\tTensorRT version: ${TensorRT_VERSION_STRING}")
endif()