#pragma once

#include "iAccurateMatcher.h"
#include "cContourBuilder.h"
#include "sim_2d_types.h"
#include "img_func.h"


class cAccurateMatcherGPU : public iAccurateMatcher
{
public:
    cAccurateMatcherGPU();
    ~cAccurateMatcherGPU();
    void setContoursBuilder(cContoursBuilderGPU const& cbg) { m_cbg = cbg; }

    virtual void setInitialPose(cv::Vec3d rvec, cv::Vec3d tvec, cv::Mat frame);

    virtual void estimate(cv::Vec3d & rvec, cv::Vec3d & tvec, cv::Mat frame);
    void estimate(cv::Vec3d & rvec, cv::Vec3d & tvec, const ImgArray<IMGTYPE> &imgArr);

private:
    int translateShot2GPU(const ImgArray<IMGTYPE> &imgArr);

private:
    cv::Vec3d m_rot_rough;
    cv::Vec3d m_tr_rough;
    cContoursBuilderGPU m_cbg;
    // Image array on GPU.
    IMGTYPE * mapGPU;
};
