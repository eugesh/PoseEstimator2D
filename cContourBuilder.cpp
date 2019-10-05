#include "cContourBuilder.h"
#include "CUDA_common.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include <cufft.h>
#include <cuda.h>


#ifdef __cplusplus
extern "C" {
#endif

int get_shift_to_create_contour_combi_sparse(int NUM_BLOCK_X, int NUM_BLOCK_Y, int NUM_THREAD_X, int NUM_THREAD_Y,
                                             MAPTYPE*mapGPU,
                                             UINT*shiftGPU_sparse, UINT*shiftGPU, UINT*widthGPU, UINT*heightGPU, int use_shadow, int shadow_value);

int create_contour_combi_sparse(int NUM_BLOCK_X, int NUM_BLOCK_Y, int NUM_THREAD_X, int NUM_THREAD_Y,
                                 MAPTYPE*contourGPU, MAPTYPE*mapGPU,
                                 UINT*shiftGPU_sparse, UINT*shiftGPU, UINT*widthGPU, UINT*heightGPU, int use_shadow, int shadow_value);
#ifdef __cplusplus
}
#endif

/**
 * @brief cContourBuilder::cContourBuilder
 * Builds countours from dicitonary of aruco markers.
 */
cContoursBuilderGPU::cContoursBuilderGPU(std::vector<int> targetValues, std::vector<int> bckgValues)
 : m_targetValues(targetValues), m_bckgValues(bckgValues)
{
    shiftArr.push_back(0);
}

void
cContoursBuilderGPU::clearAll() {
    m_templates_vec.clear();

    if(! mapContoursArr.empty()) {
        mapContoursArr.clear();
        // delete [] mapContoursArr;
        // mapContoursArr=nullptr;
    }

    if(! widthArr.empty()) {
        widthArr.clear();
        // delete [] widthArr;
        // widthArr=nullptr;
    }

    if(! heightArr.empty()) {
        heightArr.clear();
        // delete [] heightArr;
        // heightArr=nullptr;
    }

    if(! shiftArr.empty()) {
        shiftArr.clear();
        // delete [] shiftArr;
        // shiftArr=nullptr;
    }

    // free CUDA memroy
    if(mapGPU!=nullptr) {
       cudaFree(mapGPU     );
       mapGPU=nullptr;
    }
    if(contourGPU!=nullptr) {
       cudaFree(contourGPU );
       contourGPU=nullptr;
    }
    if(heightGPU!=nullptr) {
       cudaFree(heightGPU  );
       heightGPU=nullptr;
    }
    if(widthGPU!=nullptr) {
       cudaFree(widthGPU   );
       widthGPU=nullptr;
    }

    if(shiftGPU!=nullptr) {
       cudaFree(shiftGPU   );
       shiftGPU=nullptr;
    }
}

void
cContoursBuilderGPU::initAll() {
    // mapContoursArr=nullptr;
    // widthArr=nullptr;
    // heightArr=nullptr;
    // shiftArr=nullptr;

    if(! shiftArr.empty())
        shiftArr.clear();
    shiftArr.push_back(0);


    mapGPU=nullptr;
    contourGPU=nullptr;
    shiftGPU=nullptr;
    shiftGPU_sparse=nullptr;
    widthGPU=nullptr;
    heightGPU=nullptr;
}

void
cContoursBuilderGPU::setTemplates(std::vector<cv::Mat> templates_vec) {
    m_templates_vec = templates_vec;

    // shiftArr = new UINT[templates_vec.size()];
    // shiftArr[0] = 0;
    for(size_t i = 1; i < templates_vec.size(); ++i) {
        shiftArr.push_back(shiftArr[i-1] + UINT(templates_vec[i-1].cols * templates_vec[i-1].rows));
    }
}

void
cContoursBuilderGPU::append(cv::Mat img) {
    assert(img.channels() == 1);
    m_templates_vec.push_back(img);
    shiftArr.push_back(shiftArr[shiftArr.size()-1] + UINT(m_templates_vec[shiftArr.size()-1].cols * m_templates_vec[shiftArr.size()-1].rows));
    // shiftArr.push_back(shiftArr[i-1] + UINT(templates_vec[i-1].cols * templates_vec[i-1].rows));
}

void
cContoursBuilderGPU::run() {
    // EstimateShifts();

    // create_contour_combi_sparse
}

int
cContoursBuilderGPU::EstimateShifts() {
    int UseShadow = 0;
    int SHADOW_VALUE = 1;
    int Success = 0;

    { // Create contour
        int NUM_BLOCK_X = 1;// maxH;
        int NUM_BLOCK_Y = m_templates_vec.size(); // numPhi * num_plane;//numPhi;
        int NUM_THREAD_X = 1; //maxW; //maxW*maxH;
        int NUM_THREAD_Y = 1;

        int error_create_contour = get_shift_to_create_contour_combi_sparse(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_THREAD_X, NUM_THREAD_Y,
                                                                             contourGPU,
                                                                             shiftGPU_sparse, shiftGPU, widthGPU, heightGPU, UseShadow, SHADOW_VALUE);

        if( Success !=  error_create_contour )
        {
            printf("Error:create_contour: cudaLaunch : Error message = %d\n", error_create_contour ) ;

            clearAll( );

            return ErrorCudaRun;
        }

    }

    return 0;
}

int
cContoursBuilderGPU::CreateContours() {
    int UseShadow = 0;
    int SHADOW_VALUE = 1;
    int Success = 0;

    { // Create contour
        int NUM_BLOCK_X = 1;// maxH;
        int NUM_BLOCK_Y = m_templates_vec.size(); // numPhi*num_plane;//numPhi;
        int NUM_THREAD_X = 1;//maxW;//maxW*maxH;
        int NUM_THREAD_Y = 1;
        //int error_create_contour = create_contour(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_THREAD_X, NUM_THREAD_Y,
        //                                                                             contourGPU, mapGPU,
        //                                                                         shiftGPU, widthGPU, heightGPU, UseShadow);
        int error_create_contour = create_contour_combi_sparse(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_THREAD_X, NUM_THREAD_Y,
                                                                 contourGPU, mapGPU,
                                                                 shiftGPU_sparse, shiftGPU, widthGPU, heightGPU, UseShadow, SHADOW_VALUE);

        if( Success !=  error_create_contour )
        {
            printf("Error:create_contour: cudaLaunch : Error message = %d\n", error_create_contour ) ;

            clearAll( );

            return ErrorCudaRun;
        }

    }

    return 0;
}

