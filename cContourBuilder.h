#pragma once

#include <vector>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>


//!< typedef for tamplates' values type.
#include "sim_2d_types.h"

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
    void append(cv::Mat img) ;
    //!< Generate contours for templates.
    void run();

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

private:
    void initAll();
    int allocateMemoryGPU();
    int copyMemory2GPU();
    int EstimateShifts();
    int CreateContours();

private:
    //!< Contours.
    Vector<MAPTYPE> mapContoursArr;
    //!< Array of contours' length.
    Vector<UINT> shiftArr;
    Vector<UINT> widthArr;
    Vector<UINT> heightArr;
    std::vector<int> m_targetValues;
    std::vector<int> m_bckgValues;
    std::vector<cv::Mat> m_templates_vec;
private: // Vars stored on GPU.
    MAPTYPE * contourGPU;
    MAPTYPE * mapGPU;
    UINT * shiftGPU; // ToDo: Is 2^16 enough?
    UINT * shiftGPU_sparse;
    UINT * widthGPU;
    UINT * heightGPU;
};
