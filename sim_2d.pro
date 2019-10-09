#-------------------------------------------------
#
# Project created by QtCreator 2019-03-30T13:34:53
#
#-------------------------------------------------
# Requires Qt 5.12
QT += core gui widgets network

TARGET = 2DSIM
TEMPLATE = app

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
}
unix: CONFIG += link_pkgconfig
#unix: PKGCONFIG += opencv

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
CUDASOURCES = cu_matcher.cu
LIBS += -L/usr/local/cuda/lib64 -lcudart

#SET(CUDA_include_DIR /usr/local/cuda/include)

cu.output = ${QMAKE_FILE_BASE}.o
cu.commands = /usr/local/cuda/bin/nvcc -c ${QMAKE_FILE_NAME} -o ${QMAKE_FILE_OUT}
cu.input = CUDASOURCES
cu.CONFIG += no_link
cu.variable_out = OBJECTS

QMAKE_EXTRA_COMPILERS += cu

INCLUDEPATH += /usr/local/cuda/include
### CUDA end ###

SOURCES += \
    arucomatcher2d.cpp \
    cContourBuilder.cpp \
    cudaWrapper.cpp \
    i_ContoursMatcher.cpp \
    mainwindow.cpp \
    #opticalflowmatcher2d.cpp
    mat_qimage.cpp \
    sim_2d.cpp \
    main.cpp

HEADERS += \
    arucomatcher2d.h \
    cContourBuilder.h \
    cudaWrapper.h \
    i_ContoursMatcher.h \
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
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
