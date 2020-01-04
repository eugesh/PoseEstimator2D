#-------------------------------------------------
#
# Project created by QtCreator 2019-03-30T13:34:53
#
#-------------------------------------------------
# Requires Qt 5.12
QT += core gui widgets network

TARGET = 2DSIM
TEMPLATE = app

CONFIG += c++14

unix: OpenCV_DIR=/home/evgeny/soft/opencv/opencv
unix: OpenCV_contrib_DIR=/home/evgeny/soft/opencv/opencv_contrib/

win32: OpenCV_DIR=/home/evgeny/soft/opencv/opencv
win32: OpenCV_contrib_DIR=/home/evgeny/soft/opencv/opencv_contrib/

unix {
# OpenCV: Modules & Core
INCLUDEPATH = /usr/local/include/opencv2/
INCLUDEPATH += /home/evgeny/soft/opencv/opencv/include/opencv2/
INCLUDEPATH += /home/evgeny/soft/opencv/opencv/include/
INCLUDEPATH += /home/evgeny/soft/opencv/opencv/modules
INCLUDEPATH += /home/evgeny/soft/opencv/opencv/modules/video/include
INCLUDEPATH += /home/evgeny/soft/opencv/opencv/modules/videoio/include
INCLUDEPATH += /home/evgeny/soft/opencv/opencv/modules/core/include
INCLUDEPATH += /home/evgeny/soft/opencv/opencv/modules/imgproc/include
INCLUDEPATH += /home/evgeny/soft/opencv/opencv/modules/calib3d/include
INCLUDEPATH += /home/evgeny/soft/opencv/opencv/modules/features2d/include
INCLUDEPATH += /home/evgeny/soft/opencv/opencv/modules/flann/include
INCLUDEPATH += /home/evgeny/soft/opencv/opencv/modules/dnn/include
INCLUDEPATH += /home/evgeny/soft/opencv/opencv/modules/highgui/include
INCLUDEPATH += /home/evgeny/soft/opencv/opencv/modules/imgcodecs/include
INCLUDEPATH += /home/evgeny/soft/opencv/opencv/modules/ml/include
INCLUDEPATH += /home/evgeny/soft/opencv/opencv/modules/objdetect/include
INCLUDEPATH += /home/evgeny/soft/opencv/opencv/modules/photo/include
INCLUDEPATH += /home/evgeny/soft/opencv/opencv/modules/stitching/include
# OpenCV: Contrib
INCLUDEPATH += /home/evgeny/soft/opencv/opencv_contrib/modules/aruco/include
INCLUDEPATH += /home/evgeny/soft/opencv/opencv_contrib/modules/shape/include
INCLUDEPATH += /home/evgeny/soft/opencv/opencv_contrib/modules/superres/include
INCLUDEPATH += /home/evgeny/soft/opencv/opencv_contrib/modules/videostab/include
INCLUDEPATH += /home/evgeny/soft/opencv/opencv/release/
LIBS += -L/usr/local/lib -lopencv_stitching -lopencv_superres -lopencv_video -lopencv_videoio -lopencv_videostab -lopencv_imgproc -lopencv_core -lopencv_aruco -lopencv_calib3d -lopencv_highgui -lopencv_imgcodecs# -lopencv_contrib -lopencv
CONFIG += link_pkgconfig
}

win32 {
    OPENCV_DIR = "D:/opencv/ocv_4_1/msvc15/Install/opencv"
    OPENCV_INCLUDE_DIR = $$OPENCV_DIR/include/opencv2
    INCLUDEPATH += $$OPENCV_DIR/include
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR/aruco
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR/calib3d
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR/core
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR/dnn
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR/features2d
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR/flann
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR/highgui
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR/imgcodecs
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR/imgproc
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR/ml
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR/objdetect
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR/photo
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR/shape
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR/stitching
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR/superres
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR/video
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR/videoio
    INCLUDEPATH += $$OPENCV_INCLUDE_DIR/videostab
    #LIBS += D:/soft/developerTools/OpenCV/opencv_2_4_3/build/x64/vc10/bin
    #LIBS += D:/opencv/ocv_4_1/Install/opencv/bin
    #LIBS += D:/opencv/ocv_4_1/Install/opencv/x64/vc15/lib
}

#D:\opencv\ocv_4_1\Install\opencv\x64\vc15\lib
OPENCV_BUILD_DIR=D:/opencv/ocv_4_1/msvc15/Build

win32:CONFIG(release, debug|release): LIBS += -L$$OPENCV_BUILD_DIR/opencv/lib/Release -lopencv_world420
else:win32:CONFIG(debug, debug|release): LIBS += -L$$OPENCV_BUILD_DIR/opencv/lib/Debug -lopencv_world420d


# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

### CUDA ###
# Copy from https://declanrussell.com/2016/02/10/qt-cuda-and-windows-development/
CUDASOURCES = cu_matcher.cu

# Path to cuda SDK install
linux:CUDA_DIR = /usr/local/cuda
win32:CUDA_DIR = "C:\CUDA\9.2\toolkit"
# Path to cuda toolkit install
linux:CUDA_SDK = /usr/local/cuda/samples
win32:CUDA_SDK = "C:\CUDA\9.2\samples"

#Cuda include paths
INCLUDEPATH += $$CUDA_DIR/include
#INCLUDEPATH += $$CUDA_DIR/common/inc/
#INCLUDEPATH += $$CUDA_DIR/../shared/inc/
#To get some prewritten helper functions from NVIDIA
win32:INCLUDEPATH += $$CUDA_SDK\common\inc

#cuda libs
macx:QMAKE_LIBDIR += $$CUDA_DIR/lib
linux:QMAKE_LIBDIR += $$CUDA_DIR/lib64
win32:QMAKE_LIBDIR += $$CUDA_DIR\lib\x64
linux|macx:QMAKE_LIBDIR += $$CUDA_SDK/common/lib
win32:QMAKE_LIBDIR +=$$CUDA_SDK\common\lib\x64
LIBS += -lcudart -lcudadevrt

# join the includes in a line
CUDA_INC = $$join(INCLUDEPATH,'" -I"','-I"','"')

# nvcc flags (ptxas option verbose is always useful)
NVCCFLAGS = --compiler-options  -fno-strict-aliasing --ptxas-options=-v -maxrregcount 20 --use_fast_math

#On windows we must define if we are in debug mode or not
CONFIG(debug, debug|release) {
#DEBUG
    # MSVCRT link option (static or dynamic, it must be the same with your Qt SDK link option)
    win32:MSVCRT_LINK_FLAG_DEBUG = "/MDd"
    win32:NVCCFLAGS += -D_DEBUG -Xcompiler $$MSVCRT_LINK_FLAG_DEBUG
}
else{
#Release UNTESTED!!!
    win32:MSVCRT_LINK_FLAG_RELEASE = "/MD"
    win32:NVCCFLAGS += -Xcompiler $$MSVCRT_LINK_FLAG_RELEASE
}

#prepare intermediat cuda compiler
cudaIntr.input = CUDA_SOURCES
cudaIntr.output = ${OBJECTS_DIR}${QMAKE_FILE_BASE}.o
#So in windows object files have to be named with the .obj suffix instead of just .o
#God I hate you windows!!
win32:cudaIntr.output = $$OBJECTS_DIR/${QMAKE_FILE_BASE}.obj

## Tweak arch according to your hw's compute capability
cudaIntr.commands = $$CUDA_DIR/bin/nvcc -m64 -g -gencode $$GENCODE -dc $$NVCCFLAGS $$CUDA_INC $$LIBS ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}

#Set our variable out. These obj files need to be used to create the link obj file
#and used in our final gcc compilation
cudaIntr.variable_out = CUDA_OBJ
cudaIntr.variable_out += OBJECTS
cudaIntr.clean = cudaIntrObj/*.o
win32:cudaIntr.clean = cudaIntrObj/*.obj

QMAKE_EXTRA_UNIX_COMPILERS += cudaIntr

# Prepare the linking compiler step
cuda.input = CUDA_OBJ
cuda.output = ${QMAKE_FILE_BASE}_link.o
win32:cuda.output = ${QMAKE_FILE_BASE}_link.obj

# Tweak arch according to your hw's compute capability
cuda.commands = $$CUDA_DIR/bin/nvcc -m64 -g -gencode $$GENCODE  -dlink    ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cuda.dependency_type = TYPE_C
cuda.depend_command = $$CUDA_DIR/bin/nvcc -g -M $$CUDA_INC $$NVCCFLAGS   ${QMAKE_FILE_NAME}
# Tell Qt that we want add more stuff to the Makefile
QMAKE_EXTRA_UNIX_COMPILERS += cuda

#unix {
#LIBS += -L/usr/local/cuda/lib64 -lcudart
#cu.commands = /usr/local/cuda/bin/nvcc -c ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
#INCLUDEPATH += /usr/local/cuda/include
#}

#WIN32 {
#debug: {
#    LIBS += -LC:/CUDA/7_5/bin -lcudart64_75.dll
#}
#!debug: {
#    LIBS += -LC:/CUDA/7_5/bin -lcudart32_75.dll
#}
#cu.commands = C:/CUDA/7_5/bin/nvcc.exe -c ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
#INCLUDEPATH += C:/CUDA/7_5/include
#}
#SET(CUDA_include_DIR /usr/local/cuda/include)

#cu.output = ${QMAKE_FILE_BASE}.o

#cu.input = CUDASOURCES
#cu.CONFIG += no_link
#cu.variable_out = OBJECTS

#QMAKE_EXTRA_COMPILERS += cu

### CUDA end ###

SOURCES += \
    cAccurateMatcherGPU.cpp \
    cArucoMatcher2d.cpp \
    cContourBuilder.cpp \
    cudaWrapper.cpp \
    iAccurateMatcher.cpp \
    i_ContoursMatcher.cpp \
    mainwindow.cpp \
    #opticalflowmatcher2d.cpp
    mat_qimage.cpp \
    sim_2d.cpp \
    main.cpp

HEADERS += \
    cAccurateMatcherGPU.h \
    cArucoMatcher2d.h \
    cContourBuilder.h \
    camera_param.hpp \
    cudaWrapper.h \
    iAccurateMatcher.h \
    i_ContoursMatcher.h \
    img_func.h \
    mainwindow.h \
    #opticalflowmatcher2d.cpp
    mat_qimage.hpp \
    sim_2d.h \
    ehmath.hpp \
    cv_math.hpp \
    qt_math.hpp \
    CUDA_common.hpp \
    sim_2d_types.h

FORMS += \
    mainwindow.ui

# Default rules for deployment.
#qnx: target.path = /tmp/$${TARGET}/bin
#else: unix:!android: target.path = /opt/$${TARGET}/bin
#!isEmpty(target.path): INSTALLS += target

DISTFILES += \
    CMakeLists.txt
