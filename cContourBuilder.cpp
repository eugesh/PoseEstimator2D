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

}

void
cContoursBuilderGPU::clearAll() {
    m_templates_vec.clear();

    if(mapContoursArr) {
        delete [] mapContoursArr;
        mapContoursArr=nullptr;
    }

    if(widthArr) {
        delete [] widthArr;
        widthArr=nullptr;
    }

    if(heightArr) {
        delete [] heightArr;
        heightArr=nullptr;
    }

    if(shiftArr) {
        delete [] shiftArr;
        shiftArr=nullptr;
    }

    // free CUDA memroy
    if(widthGPU) {
       cudaFree(widthGPU);
       widthGPU=nullptr;
    }

    if(heightGPU) {
       cudaFree(heightGPU);
       heightGPU=nullptr;
    }
}

void
cContoursBuilderGPU::initAll() {
    mapContoursArr=nullptr;
    widthArr=nullptr;
    heightArr=nullptr;
    shiftArr=nullptr;

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

    shiftArr = new UINT[templates_vec.size()];
    shiftArr[0] = 0;
    for(int i=1; i < templates_vec.size(); ++i) {
        shiftArr[i] = shiftArr[i-1] + templates_vec[i-1].cols * templates_vec[i-1].rows;
    }
}

void
cContoursBuilderGPU::append(cv::Mat img) {
    assert(img.channels() == 1); m_templates_vec.push_back(img);
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

            clearMemory( );

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

            clearMemory( );

            return ErrorCudaRun;
        }

    }

    return 0;
}

void cContoursBuilderGPU::clearMemory( ) {
    if(mapContoursArr != nullptr) {
       free( mapContoursArr );
       mapContoursArr=nullptr;
    }
    if(widthArr!=nullptr) {
       free( widthArr );
       widthArr=nullptr;
    }
    if(heightArr!=nullptr) {
       free( heightArr );
       heightArr=nullptr;
    }
    if(shiftArr!=nullptr) {
       free( shiftArr );
       shiftArr=nullptr;
    }

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
    /*if(tableGPU!=nullptr) {
       cudaFree(tableGPU   );
       tableGPU=nullptr;
    puts("CAccurateClassifier::clearMemory, tableGPU => memory has been cleared\n");
    }

    if(lenghtContourGPU!=nullptr) {
       cudaFree(lenghtContourGPU);
       lenghtContourGPU=nullptr;
    }*/
    puts("CAccurateClassifier::clearMemory => memory has been cleared\n");
}
