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

#define KERNEL_LAUNCH 1
#define IMGTYPE float //float or double (float ������� �� CUDA)//��� ������ �����������
#define MAPTYPE unsigned int // int
#define MATHTYPE float
#define BLOCK_SIZE 512
typedef unsigned int UINT;


#ifdef __cplusplus
extern "C" {
#endif
//---- EXTERN C ---->

// User API functions.
int get_shift_to_create_contour_combi_sparse(int NUM_BLOCK_X, int NUM_BLOCK_Y, int NUM_THREAD_X, int NUM_THREAD_Y,
                                             MAPTYPE *mapGPU,
                                             UINT *shiftGPU_sparse, UINT *shiftGPU, UINT *widthGPU, UINT *heightGPU, int use_shadow, int shadow_value);

int create_contour_combi_sparse(int NUM_BLOCK_X, int NUM_BLOCK_Y, int NUM_THREAD_X, int NUM_THREAD_Y,
                                MAPTYPE *contourGPU, MAPTYPE *mapGPU,
                                UINT *shiftGPU_sparse, UINT *shiftGPU, UINT *widthGPU, UINT *heightGPU, int use_shadow, int shadow_value);

int create_matr_of_means_sparse(int NUM_BLOCK_X, int NUM_BLOCK_Y,
                                int NUM_THREAD_X, int NUM_THREAD_Y,
                                IMGTYPE *pImg, MAPTYPE *arMap,
                                int *shiftAr_sparse_cumsum, // int *lengthAr_sparse,
                                int *arW, int *arH,
                                int *shiftAr, float *pTable,
                                int ImgW, int ImgH,
                                float betta, int RX,
                                int RY, int isadd, MAPTYPE value1, MAPTYPE value2, float alpha);

// int applyAffineTransformations ();

// Pure cu functions.
// Contours creation.
__global__ void create_matr_dispersion_for_contour(IMGTYPE *pImg, MAPTYPE *arMap,
                                                   int *arW, int *arH, int *shiftAr, float *pTable, int ImgW,int ImgH, float alfa, int RX, int RY);
__global__ void get_shift_to_create_contours(UINT *shiftAr_sparse, MAPTYPE *in, UINT *shiftAr, UINT *arW, UINT *arH, int useSh, int shadow_value) ;
__global__ void create_contour_map_combi_sparse(MAPTYPE *out_sparse, UINT *shiftAr_sparse, MAPTYPE *in, UINT *shiftAr, UINT*arW, UINT *arH, int useSh, int shadow_value) ;

// Mean value calculation.
__global__ void create_matr_of_mean_values_sparse(IMGTYPE *pImg, MAPTYPE *arMap_sparse,
                                                  int *shiftAr_sparse_cumsum, //int *lengthAr_sparse,
                                                  int *arW, int *arH,
                                                  float *pTable, int ImgW, int ImgH,
                                                  float betta, int RX, int RY, int isadd, MAPTYPE value1, MAPTYPE value2, float alpha);

// Device functions.
__device__ float device_calc_mean_sparse(IMGTYPE*pImg, MAPTYPE*pContour_sparse, int shift, int pMap_sparse_cur_size, int W, int H, int ImgW, int ImgH, int RX, int RY, int ix, float alpha);
// __device__ float device_calc_mean_contour(IMGTYPE*pImg, MAPTYPE*pMap, int W, int H, int shift, int ImgW, int ImgH, int RX, int RY);

//<---- EXTERN C ----
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

int create_matr_of_means_sparse(int NUM_BLOCK_X, int NUM_BLOCK_Y,
                                int NUM_THREAD_X, int NUM_THREAD_Y,
                                IMGTYPE *pImg, MAPTYPE *arMap_sparse,
                                int *shiftAr_sparse_cumsum, // int *lengthAr_sparse,
                                int *arW, int *arH,
                                int *shiftAr, float *pTable,
                                int ImgW, int ImgH,
                                float betta, int RX,
                                int RY, int isadd, MAPTYPE value1, MAPTYPE value2, float alpha)
{ //create_matr_of_means
    dim3 dimGrid  ( NUM_BLOCK_X , NUM_BLOCK_Y ) ;
    dim3 dimBlock ( NUM_THREAD_X, NUM_THREAD_Y ) ;

#if KERNEL_LAUNCH

    create_matr_of_mean_values_sparse<<<dimGrid, dimBlock>>>
        (pImg, arMap_sparse, shiftAr_sparse_cumsum, /*lengthAr_sparse,*/ arW, arH, pTable, ImgW, ImgH, betta, RX, RY, isadd, value1, value2, alpha);

    // cudaThreadSynchronize() ;
#else
    int object_value;
    object_value = OBJECT_VALUE;
    cudaError_t resConfigCall = cudaConfigureCall(dimGrid,dimBlock);
    if( resConfigCall != cudaSuccess )
    {
        printf("Error: create_matr_of_mean_values: cudaConfigureCall1: Error message=%s\n",
                                                   cudaGetErrorString(resConfigCall));

        return ErrorCudaRun ;
    } ;
    cudaError_t resSetupArg ;
    int offset = 0 ;

        resSetupArg=cudaSetupArgument(&pImg,sizeof(IMGTYPE*),offset);
                                                     offset+=sizeof(IMGTYPE*);
        resSetupArg=cudaSetupArgument(&arMap_sparse,sizeof(MAPTYPE*),offset);
                                                     offset+=sizeof(MAPTYPE*);
        resSetupArg=cudaSetupArgument(&shiftAr_sparse_cumsum,sizeof(int*),offset);
                                                     offset+=sizeof(int*);
        resSetupArg=cudaSetupArgument(&lengthAr_sparse,sizeof(int*),offset);
                                                     offset+=sizeof(int*);
        resSetupArg=cudaSetupArgument(&arW,sizeof(int*),offset);
                                                     offset+=sizeof(int*);
        resSetupArg=cudaSetupArgument(&arH,sizeof(int*),offset);
                                                     offset+=sizeof(int*);
        resSetupArg=cudaSetupArgument(&pTable,sizeof(float*),offset);
                                                     offset+=sizeof(float*);
        resSetupArg=cudaSetupArgument(&ImgW,sizeof(int),offset);
                                                     offset+=sizeof(int);
        resSetupArg=cudaSetupArgument(&ImgH,sizeof(int),offset);
                                                     offset+=sizeof(int);
        resSetupArg=cudaSetupArgument(&betta,sizeof(float),offset);
                                                     offset+=sizeof(float);
        resSetupArg=cudaSetupArgument(&RX,sizeof(int),offset);
                                                    offset+=sizeof(int);
        resSetupArg=cudaSetupArgument(&RY,sizeof(int),offset);
                                                     offset+=sizeof(int);
        resSetupArg=cudaSetupArgument(&isadd,sizeof(int),offset);
                                                     offset+=sizeof(int);
        resSetupArg=cudaSetupArgument(&value1,sizeof(int),offset);
                                                     offset+=sizeof(int);
        resSetupArg=cudaSetupArgument(&value2,sizeof(int),offset);
                                                     offset+=sizeof(int);
        resSetupArg=cudaSetupArgument(&alpha,sizeof(float),offset);
                                                     offset+=sizeof(float);

    if( resSetupArg != cudaSuccess )
    {
       printf("Error: create_matr_of_mean_values: cudaSetupArgument: Error message=%s\n",
                                                     cudaGetErrorString(resSetupArg));

       return ErrorCudaRun;
    } ;

    cudaError_t resLaunch;

    resLaunch = cudaLaunch("create_matr_of_mean_values3") ;

    cudaThreadSynchronize() ;
    if( resLaunch != cudaSuccess )
    {
        printf("Error: create_matr_of_mean_values: cudaLaunch : Error message = %s\n",
                                             cudaGetErrorString(resLaunch) ) ;

        return ErrorCudaRun;
    } ;
#endif

    return 0;
} //create_matr_of_means

__global__ void create_matr_of_mean_values_sparse(IMGTYPE *pImg, MAPTYPE *arMap_sparse,
                    int *shiftAr_sparse_cumsum, //int lengthAr_sparse,
                    int *arW, int *arH,
                    float *pTable, int ImgW, int ImgH,
                    float betta, int RX, int RY, int isadd, MAPTYPE value1, MAPTYPE value2, float alpha)
{
    //-- Block and thread indices
    int by = blockIdx.y ;
    int tx = threadIdx.x ;
    int bx = blockIdx.x ;
    int DBx = blockDim.x ;
    int ix = bx*DBx + tx;

    int lengthAr_sparse = shiftAr_sparse_cumsum[by + 1] - shiftAr_sparse_cumsum[by];

    pTable[by*RX*RY+ix] = pTable[by*RX*RY+ix] * isadd + betta *
        // device_calc_mean3_sparse(pImg,arMap_sparse,arW[by],arH[by],shiftAr[by],ImgW,ImgH,RX,RY,ix,value1,value2,alpha);
        device_calc_mean_sparse(pImg, arMap_sparse, shiftAr_sparse_cumsum[by], lengthAr_sparse, arW[by], arH[by], ImgW, ImgH, RX, RY, ix, alpha);
}

/**
IMGTYPE*pImg - input fragment of image
MAPTYPE*pContour_sparse - input sparse matrix of contours coordinates
int pMap_sparse_cur_size - size of pMap_sparse (num_of_cpntours_elements * 2). 2 -> each element is represented by two coordinates, (x,y)
int W -
int H,
int shift - shift, current position of map
int ImgW - size of the fragment of image
int ImgH,
int RX,
int RY,
int ix,
float alpha
*/
__device__ float device_calc_mean_sparse (IMGTYPE *pImg, MAPTYPE *pContour_sparse, int shift, int pMap_sparse_cur_size, int W, int H,
                                          int ImgW, int ImgH, int RX, int RY, int ix, float alpha)
{
    if(pMap_sparse_cur_size <= 0) {
        return 0.0f;
    }

    float ksi = 0; // sum of each value along contour or inside figure

    int shiftX, shiftY;
    shiftX = (int)((float)ImgW / 2 - (float)RX / 2 + (float)(ix % RX) - (float)W / 2);
    shiftY = (int)((float)ImgH / 2 - (float)RY / 2 + (float)(ix / RX) - (float)H / 2);
    // shiftX = (int) ((float) ImgW / 2 - (float) RX + (ix % (RX * 2 / step_r)) * step_r - (float) t_w / 2 );
    // shiftY = (int) ((float) ImgH / 2 - (float) RY + (ix / (RX * 2 / step_r)) * step_r - (float) t_h / 2 );

    for(int i=0; i < pMap_sparse_cur_size; i++) {
        int x = pContour_sparse[shift + i] % W; // coords of contours elements in map SC
        int y = pContour_sparse[shift + i] / W;
           ksi += pImg[(shiftY + y) * ImgW + shiftX + x] ;
    }

    return float(ksi) / (512 * __powf( float(pMap_sparse_cur_size / 2), alpha));
}

#if 0
__global__ void create_matr_dispersion_for_contour(IMGTYPE*pImg, MAPTYPE*arMap,
                          int*arW, int*arH, int *shiftAr, float*pTable, int ImgW, int ImgH, float alfa, int RX, int RY)
{
    //-- Block and thread indexes
    int by = blockIdx.y ;
    int tx = threadIdx.x ;

    pTable[by*RX*RY+tx] += device_calc_mean_contour(pImg, arMap, arW[by], arH[by], shiftAr[by], ImgW, ImgH, RX, RY);
}

__device__ float device_calc_dispersion(IMGTYPE*pImg, MAPTYPE*pMap, int W, int H, int shift, int ImgW, int ImgH, int RX, int RY) {
  //-- Block index
  // int by = blockIdx.y ;
  // Thread index
  int tx = threadIdx.x;

  float ksi = device_calc_mean_model(pImg, pMap, W, H, shift, ImgW, ImgH, RX, RY) ;
  float sigma = 0;
  int N = 0;

  int shiftX,shiftY;
  //shiftX = tx-int((double)tx/(double)RX)*RX;
//  shiftX = tx%RX;
//  shiftY = tx/RX;//int((double)tx/(double)RX);
  shiftX = (int)((float)ImgW/2-(float)RX/2+(float)(tx%RX)-(float)W/2);
  shiftY = (int)((float)ImgH/2-(float)RY/2+(float)(tx/RX)-(float)H/2);

  for( int i = 0 ; i < H ; i ++ ) {
    for( int j = 0 ; j < W ; j ++ ) {
    //  if( !(i >= 0 && i < H && j >= 0 && j < W ) ) continue ;
      if( pMap[shift+i*W+j] == 0 ) continue ;

      float val = float( pImg[(shiftY+i)*ImgW+shiftX+j] ) ;
      float delta2 = (val - ksi)*(val - ksi) ;

      sigma += delta2 ;
      N ++ ;

    } ;
  } ;

  return sqrt(sigma) / float(N) ;
}

/**
 * @brief device_calc_mean_contour.
 * @param pImg - current frame;
 * @param pMap - contour;
 * @param W - contour's map width;
 * @param H - contour's map height;
 * @param shift
 * @param ImgW frame width;
 * @param ImgH frame height;
 * @param RX - x in ROI;
 * @param RY - y in ROI;
 * @return - mean value of contour and gradient intersection.
 */
__device__ float device_calc_mean_contour(IMGTYPE*pImg, MAPTYPE*pMap, int W, int H, int shift, int ImgW, int ImgH, int RX, int RY) {
    //  IMGTYPE ksi = 0 ;
    float ksi = 0;
    int N = 0;
    int tx = threadIdx.x;
    int bx = blockIdx.x;
    int DBx = blockDim.x;
    int ix = bx * DBx + tx;
    int shiftX, shiftY;
    // shiftX = tx % RX; // shiftX = tx-int((double)tx/(double)RX)*RX;
    // shiftY = tx / RX; //int((double)tx/(double)RX);
    shiftX = (int)((float)ImgW / 2 - (float)RX / 2 + (float)(ix % RX) - (float)W / 2);
    shiftY = (int)((float)ImgH / 2 - (float)RY / 2 + (float)(ix / RX) - (float)H / 2);
    for( int i=0; i < H; i ++ ) {
        for( int j=0; j < W; j ++ ) {
            if( pMap[shift + i * W + j] != 255 ) continue ;
            ksi += pImg[(shiftY + i) * ImgW + shiftX + j] ;
            N++;
        }
    }

    return 1.0 - float(ksi) / (float(N) * (float)255) ;
}

__device__ float device_calc_mean_model(IMGTYPE*pImg,MAPTYPE*pMap,int W,int H,int shift,int ImgW,int ImgH,int RX,int RY) {

//  IMGTYPE ksi = 0 ;
  float ksi=0;
  int N = 0 ;
  int tx = threadIdx.x ;
  int bx = blockIdx.x ;
  int DBx = blockDim.x ;
  int ix = bx*DBx + tx;
  int shiftX,shiftY ;
//  shiftX = tx%RX;//shiftX = tx-int((double)tx/(double)RX)*RX;
  //shiftY = tx/RX;//int((double)tx/(double)RX);
  shiftX = (int)((float)ImgW/2-(float)RX/2+(float)(ix%RX)-(float)W/2);
  shiftY = (int)((float)ImgH/2-(float)RY/2+(float)(ix/RX)-(float)H/2);
  for( int i = 0 ; i < H ; i ++ ) {
    for( int j = 0 ; j < W ; j ++ ) {

    //  if( !(i >= 0 && i < H && j >= 0 && j < W ) ) continue ;
      if( pMap[shift+i*W+j] != 255 ) continue ;
    /*  if( pImg[(shiftY+i)*ImgW+shiftX+j] > 200 ) {
        N ++;
        ksi += 255 ;
        continue ;
      }
      if( pImg[(shiftY+i)*ImgW+shiftX+j] <= 120&&pImg[(shiftY+i)*ImgW+shiftX+j]>50){
        N ++;
        ksi += 50 ;
        continue ;
      }
      if( pImg[(shiftY+i)*ImgW+shiftX+j] <= 50 ) {
        N ++;
        ksi += 0.001 ;
        continue ;
      } */

     // ksi += pImg[(i+shiftY)*W+j+shiftX] ;
     // ksi += pImg[(shiftY - RY/2+ImgH/2-H/2)*W+shiftX-RX/2+ImgW/2-W/2] ;
      ksi += pImg[(shiftY+i)*ImgW+shiftX+j] ;
      N ++ ;
    } ;
  } ;

  return 1.0-float(ksi) / (float(N)*(float)255) ;
}
#endif
