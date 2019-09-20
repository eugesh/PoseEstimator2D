#ifndef ARUCOMATCHER2D_H
#define ARUCOMATCHER2D_H

#include "sim_2d.h"
//#include "opencv2/opencv.hpp"
#include <opencv2/video/video.hpp>
// #include <opencv2/video/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <QObject>


class ArucoMatcher2D : public SIM_2D
{
    Q_OBJECT
public:
    explicit ArucoMatcher2D(QObject * parent = nullptr);
    virtual ~ArucoMatcher2D();

    // To be possible to implement it with QThread.
    virtual void run();

private:
    void init_contours();
    // x, y, phi relative to image SC.
    virtual QVector3D estimate_pose(cv::Mat frame);
private:
    static const int def_dict = cv::aruco::DICT_7X7_50;
    cv::Ptr<cv::aruco::Dictionary> dictionary;
    constexpr static float Marker_size = 0.1f; // [m]
};


#endif // ARUCOMATCHER2D_H
