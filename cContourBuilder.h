#pragma once

#include <vector>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>


//!< typedef for tamplates' values type.
typedef float mapType;

class cContoursBuilder
{
public:
    cContoursBuilder(std::vector<int> targetValues={1}, std::vector<int> bckgValues={0});
    //!< Set vector of binary(bw) templates' matrices.
    void setTemplates(std::vector<cv::Mat> templates_vec);
    //!< Generate contours for templates.
    void run();

public:
    // Acces funcs for resulting arrays.
    mapType * getSparseContours() { return mapContoursArr; }
    float * getShiftsArr() { return shiftArr; }
    float * getWArr() { return widthArr; }
    float * getHArr() { return heightArr; }

private:
    void EstimateShifts();
    void CreatContours();

private:
    //!< Contours.
    mapType * mapContoursArr;
    //!< Array of contours' length.
    float * shiftArr;
    float * widthArr;
    float * heightArr;
    std::vector<int> m_targetValues;
    std::vector<int> m_bckgValues;
};
