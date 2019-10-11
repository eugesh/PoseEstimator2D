#ifndef SIM_2D_H
#define SIM_2D_H

#include <QVector3D>
#include <QObject>
#include "opencv2/opencv.hpp"


// Abstract class for 2d pose estimator.
class SIM_2D : public QObject
{
        Q_OBJECT
public:
    explicit SIM_2D(QObject * parent=nullptr) { Q_UNUSED(parent) }
    virtual ~SIM_2D() { }

signals:
    void quat_raw(cv::Mat quat); // Estimated pose (Aruco marker finder).
    void quat_accurate(cv::Mat quat); // Accurately estimated (captured and matched).

public:
    // x, y, phi relative to image SC.
    virtual QVector3D estimate_pose(cv::Mat frame)=0;
    // To be possible to implement it with QThread.
    virtual void run()=0;
};

#endif // SIM_2D_H
