#include "cufft.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "math.h"
// #include "cu_touch.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include <cufft.h>
#include <cuda.h>

#define  KERNEL_LAUNCH 1
#define IMGTYPE float //float or double (float ������� �� CUDA)//��� ������ �����������
#define MAPTYPE uint16_t // int
#define MATHTYPE float
#define BLOCK_SIZE 512
typedef uint16_t UINT;


#ifdef __cplusplus
extern "C" {
#endif
// User API functions.
int get_shift_to_create_contour_combi_sparse(int NUM_BLOCK_X, int NUM_BLOCK_Y, int NUM_THREAD_X, int NUM_THREAD_Y,
                                             MAPTYPE*mapGPU,
                                             UINT*shiftGPU_sparse, UINT*shiftGPU, UINT*widthGPU, UINT*heightGPU, int use_shadow, int shadow_value);

int create_contour_combi_sparse(int NUM_BLOCK_X, int NUM_BLOCK_Y, int NUM_THREAD_X, int NUM_THREAD_Y,
                                 MAPTYPE*contourGPU, MAPTYPE*mapGPU,
                                 UINT*shiftGPU_sparse, UINT*shiftGPU, UINT*widthGPU, UINT*heightGPU, int use_shadow, int shadow_value);

// int applyAffineTransformations ();

// Pure cu functions.
__global__ void get_shift_to_create_contours(UINT*shiftAr_sparse, MAPTYPE*in, UINT*shiftAr, UINT*arW, UINT*arH,int useSh, int shadow_value) ;
__global__ void create_contour_map_combi_sparse(MAPTYPE*out_sparse, UINT*shiftAr_sparse, MAPTYPE*in, UINT*shiftAr, UINT*arW, UINT*arH, int useSh, int shadow_value) ;
#ifdef __cplusplus
}
#endif


int get_shift_to_create_contour_combi_sparse(int NUM_BLOCK_X, int NUM_BLOCK_Y, int NUM_THREAD_X, int NUM_THREAD_Y,
                                             MAPTYPE*mapGPU,
                                             UINT*shiftGPU_sparse, UINT*shiftGPU, UINT*widthGPU, UINT*heightGPU, int use_shadow, int shadow_value ) {
    dim3 dimGrid  ( NUM_BLOCK_X , NUM_BLOCK_Y  ) ;
    dim3 dimBlock ( NUM_THREAD_X, NUM_THREAD_Y ) ;

#if KERNEL_LAUNCH
    get_shift_to_create_contours<<<dimGrid, dimBlock>>>
    (shiftGPU_sparse, mapGPU, shiftGPU, widthGPU, heightGPU, use_shadow, shadow_value);

    cudaThreadSynchronize() ;
#else

    cudaError_t resConfigCall = cudaConfigureCall(dimGrid,dimBlock);
    if( resConfigCall != cudaSuccess )
    {
       printf("Error:create_contour_map_combi:cudaConfigureCall1:Error message=%s\n",
                                                   cudaGetErrorString(resConfigCall));

       return ErrorCudaRun;
    } ;
    cudaError_t resSetupArg ;
    int offset = 0 ;

    resSetupArg=cudaSetupArgument(&shiftGPU_sparse, sizeof(int*),offset);
       offset+=sizeof(int*);
    resSetupArg=cudaSetupArgument(&mapGPU, sizeof(MAPTYPE*),offset);
       offset+=sizeof(MAPTYPE*);
    resSetupArg=cudaSetupArgument(&shiftGPU, sizeof(UINT*),offset);
       offset+=sizeof(int*);
    resSetupArg=cudaSetupArgument(&widthGPU, sizeof(UINT*),offset);
       offset+=sizeof(int*);
    resSetupArg=cudaSetupArgument(&heightGPU, sizeof(UINT*),offset);
       offset+=sizeof(int*);
    resSetupArg=cudaSetupArgument(&use_shadow, sizeof(int*),offset);
       offset+=sizeof(int);
    resSetupArg=cudaSetupArgument(&shadow_value, sizeof(int*),offset);
       offset+=sizeof(int);

    if( resSetupArg != cudaSuccess )
    {
       printf("Error:create_contour_map_combi:cudaSetupArgument:Error message=%s\n",
                                                     cudaGetErrorString(resSetupArg));

       return ErrorCudaRun;
    } ;

    cudaError_t resLaunch;

    resLaunch = cudaLaunch("get_shift_to_create_contours") ;


    cudaThreadSynchronize() ;

    if( resLaunch != cudaSuccess )
    {
       printf("Error:get_shift_to_create_contours: cudaLaunch : Error message = %s\n",
                                             cudaGetErrorString(resLaunch) ) ;

       return ErrorCudaRun;
    } ;
#endif
   return 0;
}

int create_contour_combi_sparse(int NUM_BLOCK_X, int NUM_BLOCK_Y, int NUM_THREAD_X, int NUM_THREAD_Y,
                                                     MAPTYPE*contourGPU_sparse, MAPTYPE*mapGPU,
                                                     UINT*shiftGPU_sparse, UINT*shiftGPU, UINT*widthGPU, UINT*heightGPU, int use_shadow, int shadow_value )
{
    dim3 dimGrid  ( NUM_BLOCK_X , NUM_BLOCK_Y  ) ;
    dim3 dimBlock ( NUM_THREAD_X, NUM_THREAD_Y ) ;

#if KERNEL_LAUNCH

    create_contour_map_combi_sparse<<<dimGrid, dimBlock>>>
    (contourGPU_sparse, shiftGPU_sparse, mapGPU, shiftGPU, widthGPU, heightGPU, use_shadow, shadow_value);

    cudaThreadSynchronize() ;
#else

    cudaError_t resConfigCall = cudaConfigureCall(dimGrid,dimBlock);
    if( resConfigCall != cudaSuccess )
    {
       printf("Error:create_contour_map_combi:cudaConfigureCall1:Error message=%s\n",
                                                   cudaGetErrorString(resConfigCall));

       return ErrorCudaRun;
    } ;
    cudaError_t resSetupArg ;
    int offset = 0 ;

    resSetupArg=cudaSetupArgument(&contourGPU_sparse, sizeof(MAPTYPE*),offset);
       offset+=sizeof(MAPTYPE*);
    resSetupArg=cudaSetupArgument(&shiftGPU_sparse, sizeof(int*),offset);
       offset+=sizeof(int*);
    resSetupArg=cudaSetupArgument(&mapGPU, sizeof(MAPTYPE*),offset);
       offset+=sizeof(MAPTYPE*);
    resSetupArg=cudaSetupArgument(&shiftGPU, sizeof(UINT*),offset);
       offset+=sizeof(int*);
    resSetupArg=cudaSetupArgument(&widthGPU, sizeof(UINT*),offset);
       offset+=sizeof(int*);
    resSetupArg=cudaSetupArgument(&heightGPU, sizeof(UINT*),offset);
       offset+=sizeof(int*);
    resSetupArg=cudaSetupArgument(&use_shadow, sizeof(int*),offset);
       offset+=sizeof(int);
    resSetupArg=cudaSetupArgument(&shadow_value, sizeof(int*),offset);
       offset+=sizeof(int);

    if( resSetupArg != cudaSuccess )
    {
       printf("Error:create_contour_map_combi:cudaSetupArgument:Error message=%s\n",
                                                     cudaGetErrorString(resSetupArg));

       return ErrorCudaRun;
    } ;

    cudaError_t resLaunch;

    resLaunch = cudaLaunch("create_contour_map_combi_sparse") ;


    cudaThreadSynchronize() ;

    if( resLaunch != cudaSuccess )
    {
       printf("Error:create_contour_map_combi_sparse: cudaLaunch : Error message = %s\n",
                                             cudaGetErrorString(resLaunch) ) ;

       return ErrorCudaRun;
    } ;
#endif
   return 0;
}


/**
    First pass. For defining and setting shift values.
    int*shiftAr_sparse - output array of shifts(each contour length)
*/
__global__ void get_shift_to_create_contours(UINT*shiftAr_sparse, MAPTYPE*in, UINT*shiftAr, UINT*arW, UINT*arH, int useSh, int shadow_value) {
    //-- Block index 0 :2: 360*num_of_planes
    int by = blockIdx.y ;
    int i,j;
    MAPTYPE arg = 0;
    //lenght[by]=0;
    for(i=0;i<arH[by];++i) {
        for(j=0;j<arW[by];++j) {
            arg = in[shiftAr[by]+i*arW[by]+j];
            if(arg!=0&&arg!=(!useSh)*shadow_value&&i!=0&&j!=0&&i!=arH[by]&&j!=arW[by])
                if(((in[shiftAr[by]+i*arW[by]+j-1]!=arg)||
                    (in[shiftAr[by]+i*arW[by]+j+1]!=arg)||
                    (in[shiftAr[by]+(i+1)*arW[by]+j]!=arg)||
                    (in[shiftAr[by]+(i-1)*arW[by]+j]!=arg))) {
                        // out[shiftAr[by]+i*arW[by]+j]=arg;
                        shiftAr_sparse[by]++;  // length of contour. It will be shift for sparse representation of contours
                }
        }
    }
}

/**
Second pass. For setting contourse sparse matrices.
create_contour_map - create array of contours of maps around each non zero. For sparse representation of contours
areas. If designate "useSh" as 1 it will make contour around shadow area with "shadow_value" value. A map is a template is taken after rendering. Fills all the rest
elements by zero value.

MAPTYPE*out_sparse - output array of maps with only contour
int*shiftAr_sparse - array of shifts of maps (width*height), cumsum: 0, ... , length_of_contour[i-1] , ..., cumsum(0,...,length_of_cpntour[num_of_angle*num_of_planes-1])
MAPTYPE*in - input array of maps
int*shiftAr - shift array for contours matrix in usual representaion
int*arW - array of widths of each map
int*arH - array of heights of each map
int num - the value for contour's elements. The rest of elements will have zero value
int useSh - 0 or 1
int shadow_value - value of shadow area in input map "in". Also output value for contour.
*/
__global__ void create_contour_map_combi_sparse(MAPTYPE*out_sparse, UINT*shiftAr_sparse, MAPTYPE*in, UINT*shiftAr, UINT*arW, UINT*arH, int useSh, int shadow_value) {
    //-- Block index 0 .. 360 map
    int by = blockIdx.y ;
    int i,j;
    MAPTYPE arg = 0;
    int count = 0;
    //lenght[by]=0;
    for(i=0; i < arH[by]; ++i) {
        for(j=0; j < arW[by]; ++j) {
            // out[shiftAr[by]+i*arW[by]+j]=0;
            arg = in[shiftAr[by] + i * arW[by] + j];
            if(arg != 0 && arg != (!useSh) * shadow_value && i != 0 && j != 0&&i != arH[by]&&j != arW[by])
                if(((in[shiftAr[by] + i * arW[by] + j - 1] != arg) ||
                    (in[shiftAr[by] + i * arW[by] + j + 1] != arg) ||
                    (in[shiftAr[by] + (i + 1) * arW[by] + j] != arg) ||
                    (in[shiftAr[by] + (i - 1) * arW[by] + j] != arg))) {
                        out_sparse[shiftAr_sparse[by] + count] = j;
                        out_sparse[shiftAr_sparse[by] + count + 1] = i;
                        count += 2;
                        //lenght[by]++;
                }
        }
    }
}
