#pragma once

#include "sim_2d_types.h"
#include <opencv2/opencv.hpp>

class iAccurateMatcher
{
public:
    iAccurateMatcher();
    virtual ~iAccurateMatcher();

    virtual void setInitialPose(cv::Vec3d rvec, cv::Vec3d tvec, cv::Mat frame)=0;
    virtual void estimate(cv::Vec3d & rvec, cv::Vec3d & tvec, cv::Mat frame)=0;
};
