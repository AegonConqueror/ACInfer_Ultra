include(FindPackageHandleStandardArgs)

set(CUDA_SEARCH_PATH 
    PATHS 
    "/home/aegon/envs/cuda-12.4/include"
    "/home/aegon/envs/cuda-12.4/lib64"
)

set(CUDA_LIBS_LIST)
set(CUDA_VERSION_STRING)

find_path(
    CUDA_INCLUDE_DIRS
    NAMES cuda.h
    PATHS ${CUDA_SEARCH_PATH}
)

set(CUDA_ALL_LIBS
    cuda
    cudart
    cudnn
    cublas
    cublasLt
)

foreach(lib ${CUDA_ALL_LIBS})
    find_library(
      CUDA_${lib}_LIBRARY
      NAMES ${lib} 
      PATHS ${CUDA_SEARCH_PATH}
    )
    set(CUDA_LIBS_VARS CUDA_${lib}_LIBRARY ${CUDA_LIBS_LIST})
    list(APPEND CUDA_LIBS_LIST CUDA_${lib}_LIBRARY)
endforeach()

find_package_handle_standard_args(CUDA REQUIRED_VARS CUDA_INCLUDE_DIRS ${CUDA_LIBS_VARS})

if(CUDA_FOUND)
  foreach(lib ${CUDA_LIBS_LIST} )
    list(APPEND CUDA_LIBS ${${lib}})
  endforeach()
  message(STATUS ">>\t\tCUDA version : 12.4")
endif()