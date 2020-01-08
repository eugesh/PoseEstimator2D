#include "cContourBuilder.h"
#include "CUDA_common.hpp"
#include "mat_qimage.hpp"
#include "img_func.h"
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include <cufft.h>
#include <cuda.h>


#ifdef __cplusplus
extern "C" {
#endif

int get_shift_to_create_contour_combi_sparse(int NUM_BLOCK_X, int NUM_BLOCK_Y, int NUM_THREAD_X, int NUM_THREAD_Y,
                                             MAPTYPE*mapGPU,
                                             INT*shiftGPU_sparse, INT*shiftGPU, INT*widthGPU, INT*heightGPU, int use_shadow, int shadow_value);

int create_contour_combi_sparse(int NUM_BLOCK_X, int NUM_BLOCK_Y, int NUM_THREAD_X, int NUM_THREAD_Y,
                                 MAPTYPE*contourGPU, MAPTYPE*mapGPU,
                                 INT*shiftGPU_sparse, INT*shiftGPU, INT*widthGPU, INT*heightGPU, int use_shadow, int shadow_value);
#ifdef __cplusplus
}
#endif

/*template <typename T>
quint32
get_shift_to_create_contourCPU(T *in, UINT W, UINT H, int useSh, int shadow_value) {
    //-- Block index 0 :2: 360*num_of_planes
    int i, j;
    quint32 shift=0, shift_sparse = 0;
    MAPTYPE arg = 0;
    //lenght[by]=0;
    for(i=0; i < H; ++i) {
        for(j=0; j < W; ++j) {
            arg = in[shift + i * W + j];
            if(arg != 0 && arg != (! useSh) * shadow_value && i != 0 && j != 0 && i != H && j != W)
                if(((in[shift + i * W + j - 1] != arg) ||
                    (in[shift + i * W + j + 1] != arg) ||
                    (in[shift + (i + 1) * W + j] != arg) ||
                    (in[shift + (i - 1) * W + j] != arg))) {
                        // out[shiftAr[by]+i*arW[by]+j]=arg;
                        shift_sparse++;  // length of contour. It will be shift for sparse representation of contours
                }
        }
    }

    return shift_sparse;
}*/

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
/*template <typename T>
Contour create_contour_map_combi_sparse(T*in, UINT shift, UINT W, UINT H, int useSh, int shadow_value) {
    //-- Block index 0 .. 360 map
    int i,j;
    T arg = 0;
    int count = 0;
    //lenght[by]=0;
    for(i=0; i < H; ++i) {
        for(j=0; j < W; ++j) {
            // out[shiftAr[by]+i*arW[by]+j]=0;
            arg = in[shift + i * W + j];
            if(arg != 0 && arg != (!useSh) * shadow_value && i != 0 && j != 0&&i != arH[by]&&j != arW[by])
                if(((in[shift + i * W + j - 1] != arg) ||
                    (in[shift + i * W + j + 1] != arg) ||
                    (in[shift + (i + 1) * W + j] != arg) ||
                    (in[shift + (i - 1) * W + j] != arg))) {
                        out_sparse[shiftAr_sparse[by] + count] = j;
                        out_sparse[shiftAr_sparse[by] + count + 1] = i;
                        count += 2;
                        //lenght[by]++;
                }
        }
    }
}*/

/**
 *
 *
 * param[out] sparse_shift
 */
SparseContour
get_shift_and_create_contour(int *in, INT W, INT H, bool useSh, int shadow_value, INT & sparse_shift) {
    SparseContour sparse_contour;

    int arg = 0;
    sparse_shift = 0;

    for(int i=0; i < H; ++i) {
        for(int j=0; j < W; ++j) {
            // out[shiftAr[by]+i*arW[by]+j]=0;
            arg = in[i * W + j];
            if(arg!=0 && arg!=(! useSh)*shadow_value && i!=0 && j!=0 && i!=H && j!=W)
                if(((in[i * W + j - 1] != arg) ||
                    (in[i * W + j + 1] != arg) ||
                    (in[(i + 1) * W + j] != arg) ||
                    (in[(i - 1) * W + j] != arg))) {
                        sparse_contour.push_back(INT(i * W + j));
                        sparse_shift++;
                }
        }
    }

    return sparse_contour;
}

SparseContour
get_shift_and_create_contour(float *in, INT W, INT H, bool useSh, int shadow_value, INT & sparse_shift) {
    SparseContour sparse_contour;

    float arg = 0;
    sparse_shift = 0;

    for(int i=0; i < H; ++i) {
        for(int j=0; j < W; ++j) {
            // out[shiftAr[by]+i*arW[by]+j]=0;
            arg = in[i * W + j];
            if( float(fabs(arg)) < EPS_FLOAT && (arg - (! useSh) * shadow_value) < EPS_FLOAT && i!=0 && j!=0 && i!=H && j!=W)
                if(((in[i * W + j - 1] - arg) < EPS_FLOAT ||
                    (in[i * W + j + 1] - arg) < EPS_FLOAT ||
                    (in[(i + 1) * W + j] - arg) < EPS_FLOAT ||
                    (in[(i - 1) * W + j] - arg) < EPS_FLOAT )) {
                        sparse_contour.push_back(INT(i * W + j));
                        sparse_shift++;
                }
        }
    }

    return sparse_contour;
}


/**
 * @brief cContourBuilder::cContourBuilder
 * Builds countours from dicitonary of aruco markers.
 */
cContoursBuilderGPU::cContoursBuilderGPU(std::vector<int> targetValues, std::vector<int> bckgValues)
 : m_targetValues(targetValues), m_bckgValues(bckgValues)
{
	Zero();
    // shiftArr.push_back(0);
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

    if(! sparseShiftArr.empty()) {
        sparseShiftArr.clear();
        // delete [] shiftArr;
        // shiftArr=nullptr;
    }

    // free CUDA memroy
    if(mapGPU!=nullptr) {
       cudaFree(mapGPU);
       mapGPU=nullptr;
    }
    if(contourGPU!=nullptr) {
       cudaFree(contourGPU);
       contourGPU=nullptr;
    }
    if(heightGPU!=nullptr) {
       cudaFree(heightGPU);
       heightGPU=nullptr;
    }
    if(widthGPU!=nullptr) {
       cudaFree(widthGPU);
       widthGPU=nullptr;
    }

    if(shiftGPU!=nullptr) {
       cudaFree(shiftGPU);
       shiftGPU=nullptr;
    }
}

void 
cContoursBuilderGPU::Zero() {
	if (!sparseShiftArr.empty())
		sparseShiftArr.clear();
	
	contoursCPU_sparseArr = nullptr;
	mapGPU = nullptr;
	contourGPU = nullptr;
	shiftGPU = nullptr;
	shiftGPU_sparse = nullptr;
	widthGPU = nullptr;
	heightGPU = nullptr;
}

void
cContoursBuilderGPU::initAll() {
	Zero();

    sparseShiftArr.push_back(0);
}

void
cContoursBuilderGPU::setTemplates(std::vector<cv::Mat> templates_vec) {
    m_templates_vec = templates_vec;

    // shiftArr = new UINT[templates_vec.size()];
    // shiftArr[0] = 0;
    for(size_t i = 1; i < templates_vec.size(); ++i) {
        sparseShiftArr.push_back(sparseShiftArr[i-1] + INT(templates_vec[i-1].cols * templates_vec[i-1].rows));
    }
}

void
cContoursBuilderGPU::append(QImage const& img) {
    ImgArray<INT> img_ar(img);

    // Estimate shift for sparse contour.
    INT sparse_shift;
    SparseContour contour =
    get_shift_and_create_contour(img_ar.getArray(), INT(img.width()), INT(img.height()), false, 0, sparse_shift);

    // Append shift, w, h;
    if(sparseShiftArr.size()) {
        sparseShiftArr.push_back(sparseShiftArr.back() + sparse_shift);
    } else {
        sparseShiftArr.push_back(sparse_shift);
    }

    widthArr.push_back(INT(img.width()));
    heightArr.push_back(INT(img.height()));
    sparseContoursVec.push_back(contour);

    if(DEBUG) {
        QImage img_tmp = contour2Qimage(contour, widthArr.back(), heightArr.back());

        img_tmp.save(QString("./contours/c_%1_%2_%3.png").arg(int(sparseContoursVec.back()[0]*10)).arg(int(sparseContoursVec.back()[1]*10)).arg(int(sparseContoursVec.back()[2]*10)));
    }
}

/*void
cContoursBuilderGPU::append(cv::Mat img) {
    // assert(img.channels() == 1);
    m_templates_vec.push_back(img);
    shiftArr.push_back(shiftArr[shiftArr.size()-1] + UINT(m_templates_vec[shiftArr.size()-1].cols * m_templates_vec[shiftArr.size()-1].rows));
    // shiftArr.push_back(shiftArr[i-1] + UINT(templates_vec[i-1].cols * templates_vec[i-1].rows));
}*/

void
cContoursBuilderGPU::append(cv::Mat img) {
    // assert(img.channels() == 1);
    // Convert mat to qimage.
    QImage qimg = ocv::qt::mat_to_qimage_cpy(img);

    // Convert image to 1D array.
    ImgArray<INT> img_ar(qimg);

    // Estimate shift for sparse contour.
    INT sparse_shift;
    SparseContour contour =
    get_shift_and_create_contour(img_ar.getArray(), INT(img.cols), INT(img.rows), false, 0, sparse_shift);

    // Append shift, w, h;
    if(sparseShiftArr.size()) {
        sparseShiftArr.push_back(sparseShiftArr.back() + sparse_shift);
    } else {
        sparseShiftArr.push_back(sparse_shift);
    }

    widthArr.push_back(INT(img.cols));
    heightArr.push_back(INT(img.rows));

    // Generate sparse contour.

    // m_templates_vec.push_back(img);
    // shiftArr.push_back(shiftArr[shiftArr.size()-1] + UINT(m_templates_vec[shiftArr.size()-1].cols * m_templates_vec[shiftArr.size()-1].rows));
    // shiftArr.push_back(shiftArr[i-1] + UINT(templates_vec[i-1].cols * templates_vec[i-1].rows));
}

void
cContoursBuilderGPU::setNumOfTemplates(int N) {

}

int
cContoursBuilderGPU::EstimateShift4Template(cv::Mat img) {
    int shift = 0;


    return shift;
}

int
cContoursBuilderGPU::CreateContour4Template() {

    return 0;
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

    /*{ // Create contour
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

    }*/

    return 0;
}

int
cContoursBuilderGPU::CreateContours() {
    int UseShadow = 0;
    int SHADOW_VALUE = 1;
    int Success = 0;

    /*{ // Create contour
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

    }*/

    return 0;
}

QImage
cContoursBuilderGPU::contour2Qimage(SparseContour sparse_contour, int w, int h) {
    if(!w || !h || !sparse_contour.size())
        return QImage();

    QImage img ( QSize(w, h), QImage::Format_Indexed8 ) ;
    // QVector < QRgb > color_table ;
    // for( int k = 0 ; k <= 1 ; k++ ) color_table << qRgb( k, k, k ) ;
    // img.setColorTable( color_table ) ;
    for(int l = 0; l < 256; ++l)
        img.setColor( l, QColor(l,l,l).rgb( ) ) ;
    img.fill(Qt::black) ;


    for (size_t i=0; i < sparse_contour.size(); ++i) {
        INT x = sparse_contour[i] % INT(w);
        INT y = sparse_contour[i] / INT(w);

        img.setPixel(QPoint(x,y), 255);
    }

    return img;
}

bool
cContoursBuilderGPU::validateBuildContours(QString path) {
    bool rez = false;

    for (int i=0; i < sparseContoursVec.size(); ++i) {

    }

    return rez;
}

/**
 * @brief cContoursBuilderGPU::contoursSetup
 * Call it when all contours were appended.
 */
void
cContoursBuilderGPU::contoursSetup() {
    prepareMemoryCPU();
    allocateMemoryGPU();
    copyMemory2GPU();
}

int
cContoursBuilderGPU::prepareMemoryCPU() {
    contoursCPU_sparseArr = static_cast<INT*> (malloc(sizeof (INT) * (sparseShiftArr.back()))); // + sparseContoursVec.back().size())));

    for(size_t i = 0; i < sparseContoursVec.size(); ++i) {
        INT shift = 0;
        if(i > 0) {
            shift = sparseShiftArr[i-1];
        }
        memcpy(&contoursCPU_sparseArr[shift], sparseContoursVec[i].data(), sizeof(INT) * sparseContoursVec[i].size());
    }

    return 0;
}

int
cContoursBuilderGPU::allocateMemoryGPU() {
    cudaError_t mem2d = cudaMalloc(reinterpret_cast<void**>(&widthGPU), sizeof(INT) * widthArr.size());

    mem2d = cudaMalloc(reinterpret_cast<void**>(&heightGPU), sizeof(INT) * heightArr.size());

    mem2d = cudaMalloc(reinterpret_cast<void**>(&shiftGPU_sparse), sizeof(INT) * (sparseShiftArr.size()));

    mem2d = cudaMalloc(reinterpret_cast<void**>(&contourGPU), sizeof(MAPTYPE) * (sparseShiftArr.back())); // + sparseContoursVec.back().size()));

    return 0;
}

int
cContoursBuilderGPU::copyMemory2GPU() {
    cudaError_t mem2d = cudaMemcpy(widthGPU, widthArr.data(), sizeof(INT) * widthArr.size(), cudaMemcpyHostToDevice);

    mem2d = cudaMemcpy(heightGPU, heightArr.data(), sizeof(INT) * heightArr.size(), cudaMemcpyHostToDevice);

    mem2d = cudaMemcpy(shiftGPU_sparse, sparseShiftArr.data(), sizeof(INT) * sparseShiftArr.size(), cudaMemcpyHostToDevice);

    mem2d = cudaMemcpy(contourGPU, contoursCPU_sparseArr, sizeof(MAPTYPE) * (sparseShiftArr.back()/* + sparseContoursVec.back().size()*/), cudaMemcpyHostToDevice);

    return mem2d;
}
