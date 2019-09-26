#include "cContourBuilder.h"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include <cufft.h>
#include <cuda.h>

extern "C" {

int get_shift_to_create_contour_combi_sparse(int NUM_BLOCK_X, int NUM_BLOCK_Y, int NUM_THREAD_X, int NUM_THREAD_Y,
                                                     MAPTYPE*mapGPU,
                                                     int*shiftGPU, int*widthGPU, int*heightGPU, int use_shadow, int shadow_value);

int create_contour_combi_sparse(int NUM_BLOCK_X, int NUM_BLOCK_Y, int NUM_THREAD_X, int NUM_THREAD_Y,
                                                     MAPTYPE*contourGPU, MAPTYPE*mapGPU,
                                                     int*shiftGPU, int*widthGPU, int*heightGPU, int use_shadow, int shadow_value);

}


/**
 * @brief cContourBuilder::cContourBuilder
 * Builds countours from dicitonary of aruco markers.
 */
cContoursBuilderGPU::cContoursBuilderGPU(std::vector<int> targetValues, std::vector<int> bckgValues)
 : m_targetValues(targetValues), m_bckgValues(bckgValues)
{

}


void
cContoursBuilderGPU::run() {
    // EstimateShifts();

    // create_contour_combi_sparse
}

int
cContoursBuilderGPU::EstimateShifts() {
    int UseShadow=0;
    int SHADOW_VALUE=1;
    int Success=0;

    { // Create contour
        int NUM_BLOCK_X = 1;// maxH;
        int NUM_BLOCK_Y = m_templates_vec.size(); // numPhi * num_plane;//numPhi;
        int NUM_THREAD_X = 1;//maxW;//maxW*maxH;
        int NUM_THREAD_Y = 1;
        //int error_create_contour = create_contour(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_THREAD_X, NUM_THREAD_Y,
        //                                                                             contourGPU, mapGPU,
        //                                                                         shiftGPU, widthGPU, heightGPU, UseShadow);
        // get_shift_to_create_contour_combi_sparse();
        int error_create_contour = create_contour_combi(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_THREAD_X, NUM_THREAD_Y,
                                                                                     contourGPU, mapGPU,
                                                                                     shiftGPU, widthGPU, heightGPU, UseShadow, SHADOW_VALUE);

        if( Success !=  error_create_contour )
        {
            printf("Error:create_contour: cudaLaunch : Error message = %d\n", error_create_contour ) ;

            clearMemory( );

            return ErrorCudaRun;
        };

    }
}

int
cContoursBuilderGPU::CreateContours() {
    int Success=0;
    { // Create contour
        int NUM_BLOCK_X = 1;// maxH;
        int NUM_BLOCK_Y = numPhi*num_plane;//numPhi;
        int NUM_THREAD_X = 1;//maxW;//maxW*maxH;
        int NUM_THREAD_Y = 1;
        //int error_create_contour = create_contour(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_THREAD_X, NUM_THREAD_Y,
        //                                                                             contourGPU, mapGPU,
        //                                                                         shiftGPU, widthGPU, heightGPU, UseShadow);
        int error_create_contour = create_contour_combi(NUM_BLOCK_X, NUM_BLOCK_Y, NUM_THREAD_X, NUM_THREAD_Y,
                                                                                     contourGPU, mapGPU,
                                                                                     shiftGPU, widthGPU, heightGPU, UseShadow, SHADOW_VALUE);

        if( Success !=  error_create_contour )
        {
            printf("Error:create_contour: cudaLaunch : Error message = %d\n", error_create_contour ) ;

            clearMemory( );

            return ErrorCudaRun;
        };

    }
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
    if(tableGPU!=nullptr) {
       cudaFree(tableGPU   );
       tableGPU=nullptr;
    puts("CAccurateClassifier::clearMemory, tableGPU => memory has been cleared\n");
    }

    if(lenghtContourGPU!=nullptr) {
       cudaFree(lenghtContourGPU);
       lenghtContourGPU=nullptr;
    }
    puts("CAccurateClassifier::clearMemory => memory has been cleared\n");
}
