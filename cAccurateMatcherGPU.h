#pragma once

#include "iAccurateMatcher.h"
#include "cContourBuilder.h"


class cAccurateMatcherGPU : public iAccurateMatcher
{
public:
    cAccurateMatcherGPU();
    ~cAccurateMatcherGPU();

    void setInitialPose(cv::Vec3d rvec, cv::Vec3d tvec, cv::Mat frame);

    void estimate(cv::Vec3d & rvec, cv::Vec3d & tvec, cv::Mat frame);

private:
    void setContoursBuilder(cContoursBuilderGPU const& cbg) { m_cbg = cbg; }
    cv::Vec3d m_rot_rough;
    cv::Vec3d m_tr_rough;
    cContoursBuilderGPU m_cbg;
};
