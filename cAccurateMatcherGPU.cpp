#include "cAccurateMatcherGPU.h"

cAccurateMatcherGPU::cAccurateMatcherGPU()
{

}

cAccurateMatcherGPU::~cAccurateMatcherGPU() {

}

void
cAccurateMatcherGPU::setInitialPose(cv::Vec3d rvec, cv::Vec3d tvec, cv::Mat frame) {
    m_rot_rough = rvec;
    m_tr_rough = tvec;
}

void
cAccurateMatcherGPU::estimate(cv::Vec3d & rvec, cv::Vec3d & tvec, cv::Mat frame) {

}
