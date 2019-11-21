#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

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

void
cAccurateMatcherGPU::estimate(cv::Vec3d & rvec, cv::Vec3d & tvec, const ImgArray<IMGTYPE> & imgArr) {
    // Translate (copy) Gradient of current shot to GPU.
    translateShot2GPU(imgArr);

    // Calculate Block size and Threads per Block numbers.
    ROI_MARGIN; // Block y size
    // N_templates -> num of xThreads

    // Run matching process on GPU

    // Transfer table of mean values device->host

    // Find maximum.

    // Emit quaternion.
}

int
cAccurateMatcherGPU::translateShot2GPU(const ImgArray<IMGTYPE> &imgArr) {
    // Allocate memory.
    cudaError_t mem2d = cudaMalloc(reinterpret_cast<void**>(&mapGPU), sizeof(IMGTYPE) * imgArr.width() * imgArr.height());

    // Copy memory.
    mem2d = cudaMemcpy(mapGPU, imgArr.getArray(), sizeof(IMGTYPE) * imgArr.width() * imgArr.height(), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    return 0;
}
