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
    void setContoursBuilder(cContoursBuilderGPU const& cbg);

    virtual void setInitialPose(cv::Vec3d rvec, cv::Vec3d tvec, cv::Mat frame);

    virtual void estimate(cv::Vec3d & rvec, cv::Vec3d & tvec, cv::Mat frame);
    void estimate(cv::Vec3d & rvec, cv::Vec3d & tvec, const ImgArray<IMGTYPE> &imgArr);

private:
    int translateShot2GPU(const ImgArray<IMGTYPE> &imgArr);
    int getTableFromGPU();
    int runMatching(const ImgArray<IMGTYPE> &imgArr);
    int Clear();
    int Zero();
    int init_memory();

private:
    cv::Vec3d m_rot_rough;
    cv::Vec3d m_tr_rough;
    cContoursBuilderGPU m_cbg;
    // CPU
    float *m_pTable;
    // GPU
    // Image array on GPU.
    IMGTYPE * m_shotGPU;
    float *m_pTableGPU;

    int D_ROI_X, D_ROI_Y;
    int m_NUM_BLOCK_X;
    int m_NUM_BLOCK_Y;
    int m_NUM_THREAD_X;
    int m_NUM_THREAD_Y;

};
