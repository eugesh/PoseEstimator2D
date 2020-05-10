#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <QFile>
#include <QByteArray>

#include "cAccurateMatcherGPU.h"



#ifdef __cplusplus
extern "C" {
#endif

int create_matr_of_means_sparse(int NUM_BLOCK_X,  int NUM_BLOCK_Y,
                                int NUM_THREAD_X, int NUM_THREAD_Y,
                                IMGTYPE *pImg,    MAPTYPE *arMap,
                                INT *shiftAr_sparse_cumsum, // int *lengthAr_sparse,
                                INT *arW,         INT *arH,
                                float *pTable,
                                int ImgW,         int ImgH,
                                float betta,      int RX,
                                int RY,           int isadd,
                                MAPTYPE value1,   MAPTYPE value2,
                                float alpha);

#ifdef __cplusplus
}
#endif


cAccurateMatcherGPU::cAccurateMatcherGPU() {
    Zero();

    D_ROI_X = 2 * ROI_MARGIN / ROI_STEP;
    D_ROI_Y = 2 * ROI_MARGIN / ROI_STEP;
}

int
cAccurateMatcherGPU::Clear() {

    if(m_pTable)
        delete[] m_pTable;

    if(m_pTableGPU != nullptr)
        cudaFree(m_pTableGPU);

    if(m_shotGPU != nullptr)
        cudaFree(m_shotGPU);

    Zero();

    return 0;
}

int
cAccurateMatcherGPU::Zero() {
    m_pTable=nullptr;

    m_pTableGPU=nullptr;
    m_shotGPU=nullptr;

    return 0;
}

int
cAccurateMatcherGPU::init_memory() {
    m_pTable = new float[m_NUM_BLOCK_X * m_NUM_BLOCK_Y * m_NUM_THREAD_X];

    // Allocate memory for m_shotGPU.

    // Allocate memory for table.
    cudaError_t mem2d = cudaMalloc(reinterpret_cast<void**>(&m_pTableGPU), sizeof(float) * m_NUM_BLOCK_X * m_NUM_BLOCK_Y * m_NUM_THREAD_X);

    if (mem2d) {
        printf("Error: init_memory: cudaMalloc : Error message = %d\n", mem2d);
        // clearMemory( );
        // return ErrorCudaRun;
    }

    mem2d = cudaMemcpy(m_pTableGPU, m_pTable, sizeof(float) * m_NUM_BLOCK_Y * m_NUM_BLOCK_X * m_NUM_THREAD_X, cudaMemcpyHostToDevice);

    return mem2d;
}

void
cAccurateMatcherGPU::setContoursBuilder(cContoursBuilderGPU const& cbg) {
    m_cbg = cbg;

    Clear();

    m_NUM_BLOCK_X = m_cbg.get_sparseContoursVec().size(); // / BLOCK_SIZE;
    m_NUM_BLOCK_Y = 1; //BLOCK_SIZE;
    m_NUM_THREAD_X = int ( ceil( double( D_ROI_X * D_ROI_Y )));
    m_NUM_THREAD_Y = 1;

    init_memory();
}

cAccurateMatcherGPU::~cAccurateMatcherGPU() {

}

int
cAccurateMatcherGPU::translateShot2GPU(const ImgArray<IMGTYPE> &imgArr) {
    // Allocate memory.
    if (m_shotGPU) {
        cudaFree(m_shotGPU);
        m_shotGPU = nullptr;
    }
    cudaError_t mem2d = cudaMalloc(reinterpret_cast<void**>(&m_shotGPU), sizeof(IMGTYPE) * imgArr.width() * imgArr.height());

    // Copy memory.
    mem2d = cudaMemcpy(m_shotGPU, imgArr.getArray(), sizeof(IMGTYPE) * imgArr.width() * imgArr.height(), cudaMemcpyHostToDevice);

    if (mem2d) {
        printf("Error: translateShot2GPU: cudaMemcpy : Error message = %d\n", mem2d);
        // clearMemory( );
        // return ErrorCudaRun;
    }

    cudaDeviceSynchronize();

    return mem2d;
}

int
cAccurateMatcherGPU::getTableFromGPU() {
    cudaError_t mem2d = cudaMemcpy(m_pTable, m_pTableGPU, sizeof(float) * m_NUM_BLOCK_Y * m_NUM_BLOCK_X * m_NUM_THREAD_X, cudaMemcpyDeviceToHost);

	cudaDeviceSynchronize();

    if (mem2d) {
        printf("Error: getTableFromGPU: cudaMemcpy : Error message = %d\n", mem2d);
        // clearMemory( );
        // return ErrorCudaRun;
    }

	QFile filep("m_pTable.txt");
	if (!filep.open(QFile::WriteOnly)) {
		
	}

    for(int i=0; i < m_NUM_THREAD_X; ++i) {
        for(int j=0; j < m_NUM_BLOCK_X * m_NUM_BLOCK_Y; ++j) {
            // std::cout << m_pTable[i * m_NUM_BLOCK_X * m_NUM_BLOCK_Y + j] << std::endl;
            QByteArray arr;
            arr.append(QString("%1 ").arg(m_pTable[i * m_NUM_BLOCK_X * m_NUM_BLOCK_Y + j]));
            filep.write(arr);
        }
        // std::cout << std::endl;
        filep.write("\n");
    }

    return mem2d;
}

void
cAccurateMatcherGPU::setInitialPose(cv::Vec3d rvec, cv::Vec3d tvec, cv::Mat frame) {
    m_rot_rough = rvec;
    m_tr_rough = tvec;
}

void
cAccurateMatcherGPU::estimate(cv::Vec3d & rvec, cv::Vec3d & tvec, cv::Mat frame) {

}

int
cAccurateMatcherGPU::runMatching(const ImgArray<IMGTYPE> & imgArr) {
    // Calculate Block size and Threads per Block numbers.
    ROI_MARGIN; // Block y size
    // N_templates -> num of xThreads

    // Run matching process on GPU
    float betta = 1; // alphaContour * (-1);
    int isadd = 0;
    int value1 = 1; // OBJECT_VALUE;
    int value2 = 0; // SHADOW_VALUE;
    float alpha = 0.7;

    int error_create_matr = 0;
    error_create_matr = create_matr_of_means_sparse(m_NUM_BLOCK_X, m_NUM_BLOCK_Y, m_NUM_THREAD_X, m_NUM_THREAD_Y,
                                                    m_shotGPU, m_cbg.getSparseContoursGPU(),
                                                    m_cbg.getSparceShiftsGPU(), // int *lengthAr_sparse,
                                                    m_cbg.getWGPU(), m_cbg.getHGPU(),
                                                    m_pTableGPU,
                                                    imgArr.width(), imgArr.height(),
                                                    betta, D_ROI_X, D_ROI_Y, isadd,
                                                    value1, value2, alpha);

    cudaDeviceSynchronize();

    if(error_create_matr) {
        printf("Error: create_matr_of_means_sparse: cudaLaunch : Error message = %d\n", error_create_matr  ) ;
        // clearMemory( );
        // return ErrorCudaRun;
    }

    return error_create_matr;
}

void
cAccurateMatcherGPU::estimate(cv::Vec3d & rvec, cv::Vec3d & tvec, const ImgArray<IMGTYPE> & imgArr) {
    // Translate (copy) Gradient of current shot to GPU.
    translateShot2GPU(imgArr);

    runMatching(imgArr);

    // Transfer table of mean values device->host
    getTableFromGPU();

    // Find maximum.
    float max=0.0f;
    for(int i=0; i < m_NUM_BLOCK_Y * m_NUM_THREAD_X; ++i) {
        float val = m_pTable[i];
        if(val > max)
            max = val;
    }

    std::cout << "max value = " << max << std::endl;
    printf("max value from printf: %f", max);

    // Emit quaternion.
}
