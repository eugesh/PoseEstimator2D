#pragma once

#include <vector>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>


//!< typedef for tamplates' values type.
typedef float MAPTYPE;

class cContoursBuilderGPU
{
public:
    cContoursBuilderGPU(std::vector<int> targetValues={1}, std::vector<int> bckgValues={0});
    //!< Set vector of binary(bw) templates' matrices.
    void setTemplates(std::vector<cv::Mat> templates_vec) { m_templates_vec = templates_vec; }
    //!< Generate contours for templates.
    void run();

public:
    // Acces funcs for resulting arrays.
    MAPTYPE * getSparseContours() { return mapContoursArr; }
    float * getShiftsArr() { return shiftArr; }
    float * getWArr() { return widthArr; }
    float * getHArr() { return heightArr; }

private:
    int EstimateShifts();
    int CreateContours();
    void clearMemory();

private:
    //!< Contours.
    MAPTYPE * mapContoursArr;
    //!< Array of contours' length.
    float * shiftArr;
    float * widthArr;
    float * heightArr;
    std::vector<int> m_targetValues;
    std::vector<int> m_bckgValues;
    std::vector<cv::Mat> m_templates_vec;
private: // Vars stored on GPU.
    MAPTYPE * contourGPU;
    MAPTYPE * mapGPU;
    float * shiftGPU;
    float * widthGPU;
    float * heightGPU;
};
