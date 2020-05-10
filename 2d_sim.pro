#-------------------------------------------------
#
# Project created by QtCreator 2019-03-30T13:34:53
#
#-------------------------------------------------

QT += core gui widgets network

TARGET = 2d_sim
TEMPLATE = app

unix {
INCLUDEPATH = /usr/local/include/opencv2/
#LIBS += -L/usr/local/lib -lopencv_stitching -lopencv_superres -lopencv_contrib -lopencv
LIBS += -L/usr/local/cuda/lib64 -lcudart

CONFIG += link_pkgconfig
PKGCONFIG += opencv
}


# Set your filepaths
#if(UNIX)
#    SET(OpenCV_DIR /home/evgeny/soft/opencv/opencv/release)
#    LINK_DIRECTORIES(/usr/local/lib/)
#endif (UNIX)

WIN32 {
    #INCLUDEPATH += D:/soft/developerTools/OpenCV/opencv_2_4_3/build #(OpenCV_DIR
    INCLUDEPATH += D:/soft/developerTools/OpenCV/opencv_2_4_3/build/include
    LIBS += D:/soft/developerTools/OpenCV/opencv_2_4_3/build/x64/vc10/bin
}

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

SOURCES += \
    cArucoMatcher2d.cpp \
    cAccurateMatcherGPU.cpp \
    cContourBuilder.cpp \
    cudaWrapper.cpp \
    i_ContoursMatcher.cpp \
    mainwindow.cpp \
    #opticalflowmatcher2d.cpp \
    sim_2d.cpp \
    main.cpp

HEADERS += \
    cAccurateMatcherGPU.h \
    cArucoMatcher2d.h \
    cContourBuilder.h \
    cudaWrapper.h \
    cv_math.hpp \
    ehmath.hpp \
    i_ContoursMatcher.h \
    mainwindow.h \
    #opticalflowmatcher2d.h \
    sim_2d.h \
    cv_math.hpp

FORMS += \
        mainwindow.ui

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

#DISTFILES += \
#    gen_marker.py

DISTFILES += \
    accurate_aruco.py \
    cu_matcher.cu \
    CMakeLists.txt
