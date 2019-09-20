cmake_minimum_required(VERSION 2.8)

project(2DSIM)

find_package(PkgConfig)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

#if(NOT OpenCV_FOUND)
#    MESSAGE(FATAL_ERROR "OpenCV not found")
#endif()

set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_videoio opencv_contrib opencv_calib3d opencv_features2d opencv_flann opencv_photo)

if(CMAKE_VERSION VERSION_LESS "2.8.11")
    include_directories(${OpenCV_INCLUDE_DIRS})
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Target older architectures for those who haven't upgraded their graphics card to the latest models.
#set(CUDA_NVCC_FLAGS "-gencode arch=compute_11,code=sm_11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=compute_50 ")

cuda_add_library(2DSIM_kernels
    cu_matcher.cu
)

# Qt based
find_package(Qt5Widgets)
if(Qt5Widgets_FOUND)
    set(CMAKE_AUTOMOC ON)
    #find_package(Qt5OpenGL REQUIRED)
    include_directories(${Qt5Widgets_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})# ${Qt5OpenGL_INCLUDE_DIRS})

    add_executable(2DSIM
        arucomatcher2d.cpp
        cContourBuilder.cpp
        cudaWrapper.cpp
        i_ContoursMatcher.cpp
        mainwindow.cpp
        #opticalflowmatcher2d.cpp
        sim_2d.cpp
        main.cpp
    )

    target_link_libraries(2DSIM 2DSIM_kernels Qt5::Widgets OpenCV_LIBS)# Qt5::OpenGL ${OPENGL_LIBRARIES} )
    set_target_properties(2DSIM PROPERTIES COMPILE_FLAGS "-std=c++11")
endif(Qt5Widgets_FOUND)
