#pragma once

#include <vector>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <QString>

//!< typedef for tamplates' values type.
#include "sim_2d_types.h"

class QImage;


/**
 * @brief The cContoursBuilderGPU class
 * ContoursBuilder class, builds contours from set of images.
 *
 * ToDo: implement another class, creating contours from one image on GPU applying affine transformations realtime before performing it's work.
 */
class cContoursBuilderGPU
{
public:
    cContoursBuilderGPU(std::vector<int> targetValues={1}, std::vector<int> bckgValues={0});
    //!< Set vector of binary(bw) templates' matrices.
    void setTemplates(std::vector<cv::Mat> templates_vec) ;
    void append(cv::Mat img);
    void append(QImage const& img);
    //!< Generate contours for templates.
    void run();
    //!< Call This function if you know number of tamplates in advance. It speeds up memory allocation for internal buffers doing it only once.
    void setNumOfTemplates(int N);

public:
    // Acces funcs for resulting arrays.
    MAPTYPE * getSparseContours() { return mapContoursArr.data(); }
    UINT * getShiftsArr() { return shiftArr.data(); }
    UINT * getWArr() { return widthArr.data(); }
    UINT * getHArr() { return heightArr.data(); }
    MAPTYPE * getSparseContoursGPU() { return contourGPU; }
    UINT * getShiftsGPU() { return shiftGPU; }
    UINT * getSparceShiftsGPU() { return shiftGPU; }
    UINT * getWGPU() { return widthGPU; }
    UINT * getHGPU() { return heightGPU; }
    void clearAll();
    // Debug.
    bool validateBuildContours(QString path="");
    static QImage contour2Qimage(SparseContour sparse_contour, int w, int h);

private:
    void initAll();
    int allocateMemoryGPU();
    int copyMemory2GPU();
    int EstimateShifts();
    int EstimateShift4Template(cv::Mat img);
    int CreateContours();
    int CreateContour4Template();

private:
    //!< Contours.
    Vector<MAPTYPE> mapContoursArr;
    Vector<SparseContour> sparseContoursVec;
    //!< Array of contours' length.
    Vector<UINT> shiftArr;
    Vector<UINT> widthArr;
    Vector<UINT> heightArr;
    Vector<int> m_targetValues;
    Vector<int> m_bckgValues;
    Vector<cv::Mat> m_templates_vec;

private: // Vars stored on GPU.
    MAPTYPE * contourGPU;
    MAPTYPE * mapGPU;
    UINT * shiftGPU; // ToDo: Is 2^16 enough?
    UINT * shiftGPU_sparse;
    UINT * widthGPU;
    UINT * heightGPU;
};
