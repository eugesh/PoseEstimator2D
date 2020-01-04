cmake_minimum_required(VERSION 2.8)

project(2DSIM)

# Set your filepaths
if(UNIX)
    SET(OpenCV_DIR /home/evgeny/soft/opencv/opencv/release)
    LINK_DIRECTORIES(/usr/local/lib/)
endif (UNIX)

if (WIN32)
	SET(QT_MAKE_EXECUTABLE C:/Qt/5.12.6/msvc2017_64/bin/qmake.exe)
	SET(Qt5Core_DIR C:/Qt/5.12.6/msvc2017_64/lib/cmake/Qt5Core)
	SET(Qt5Widgets_DIR C:/Qt/5.12.6/msvc2017_64/lib/cmake/Qt5Widgets)
	SET(Qt5Gui_DIR C:/Qt/5.12.6/msvc2017_64/lib/cmake/Qt5Gui)
    set(OpenCV_DIR D:/opencv/ocv_4_1/msvc15/Build/opencv)
    SET(OpenCV_INSTALL_DIR D:/opencv/ocv_4_1/Install/opencv)
    SET(OPENCV_BUILD_DIR D:/opencv/ocv_4_1/msvc15/Build)
	SET(OPENCV_INCLUDE_DIRS ${OpenCV_INSTALL_DIR}/include/opencv2)
    #SET(OpenCV_DIR D:/soft/developerTools/OpenCV/opencv_2_4_3/build/x86/vc9)
	if(CMAKE_BUILD_TYPE MATCHES Release)
		LINK_DIRECTORIES(${OpenCV_DIR}/lib/Release)
	else()
		LINK_DIRECTORIES(${OpenCV_DIR}/lib/Debug)
	endif()
endif (WIN32)

include_directories(${OpenCV_INSTALL_DIR}/include)
include_directories(${OpenCV_INCLUDE_DIRS} ${OPENCV_BUILD_DIR}/opencv)

find_package(PkgConfig)
find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

if(NOT OpenCV_FOUND)
    MESSAGE(FATAL_ERROR "OpenCV not found")
endif()

set(OpenCV_LIBS opencv_core opencv_highgui opencv_imgproc opencv_videoio opencv_calib3d opencv_features2d opencv_flann opencv_photo opencv_aruco) #opencv_contrib

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
# Requires Qt 5.12
find_package(Qt5Widgets)
if(Qt5Widgets_FOUND)
    set(CMAKE_AUTOMOC ON)
    QT5_WRAP_UI(UISrcs ${CMAKE_CURRENT_SOURCE_DIR}/mainwindow.ui)
    #find_package(Qt5OpenGL REQUIRED)
    include_directories(${Qt5Widgets_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS} ${OPENCV_BUILD_DIR}/opencv)# ${Qt5OpenGL_INCLUDE_DIRS})

    add_executable(2DSIM
        cAccurateMatcherGPU.cpp
        cArucoMatcher2d.cpp
        cContourBuilder.cpp
        cudaWrapper.cpp
        iAccurateMatcher.cpp
        i_ContoursMatcher.cpp
        mainwindow.cpp
        #opticalflowmatcher2d.cpp
        mat_qimage.cpp
        sim_2d.cpp
        main.cpp
        ${UISrcs}
    )

if (UNIX)
    target_link_libraries(2DSIM 2DSIM_kernels Qt5::Widgets ${OpenCV_LIBS} stdc++ m)# Qt5::OpenGL ${OPENGL_LIBRARIES} )
endif()

if(WIN32)
	target_link_libraries(2DSIM 2DSIM_kernels Qt5::Widgets ${OpenCV_LIBS})
endif()

    set_target_properties(2DSIM PROPERTIES COMPILE_FLAGS "-std=c++11")
endif(Qt5Widgets_FOUND)
