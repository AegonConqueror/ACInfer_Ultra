include(FindPackageHandleStandardArgs)

set(OpenCV_SEARCH_PATH 
    PATHS 
    "/home/aegon/envs/opencv/include/opencv4"
    "/home/aegon/envs/opencv/lib"
    "/home/aegon/envs/opencv/lib/cmake/opencv4"
)

set(OpenCV_LIBS_LIST)
set(OpenCV_VERSION_STRING)

find_path(
    OpenCV_INCLUDE_DIRS
    NAMES opencv2/core.hpp
    PATHS ${OpenCV_SEARCH_PATH}
    NO_DEFAULT_PATH
)

set(OpenCV_ALL_LIBS
    opencv_core
    opencv_highgui
    opencv_imgcodecs
    opencv_imgproc
    opencv_video
    opencv_videoio
    opencv_calib3d
    opencv_dnn
)

foreach(lib ${OpenCV_ALL_LIBS})
    find_library(
      OpenCV_${lib}_LIBRARY
      NAMES ${lib} 
      PATHS ${OpenCV_SEARCH_PATH}
      NO_DEFAULT_PATH
    )
    set(OpenCV_LIBS_VARS OpenCV_${lib}_LIBRARY ${OpenCV_LIBS_LIST})
    list(APPEND OpenCV_LIBS_LIST OpenCV_${lib}_LIBRARY)
endforeach()


find_file(OpenCV_VERSION_FILE
    NAMES
    OpenCVConfig-version.cmake
    PATHS ${OpenCV_SEARCH_PATH}
  )

if(OpenCV_VERSION_FILE)
  include(${OpenCV_VERSION_FILE})
  string(REGEX MATCH "[0-9]+\\.[0-9]+\\.[0-9]+" OpenCV_VERSION_MATCH ${OpenCV_VERSION})
  if(NOT ${OpenCV_VERSION_STRING} STREQUAL ${OpenCV_VERSION_MATCH})
    message(FATAL_ERROR "Found OpenCV version ${OpenCV_VERSION_MATCH}, but required version is ${OpenCV_VERSION_STRING}")
  endif()
else()
  message(FATAL_ERROR "OpenCV version file not found")
endif()

find_package_handle_standard_args(OpenCV REQUIRED_VARS OpenCV_INCLUDE_DIRS ${OpenCV_LIBS_VARS})

if(OpenCV_FOUND)
  foreach(lib ${OpenCV_LIBS_LIST} )
    list(APPEND OpenCV_LIBS ${${lib}})
  endforeach()
  message(STATUS ">>\t\tOpenCV version: ${OpenCV_VERSION}")
endif()