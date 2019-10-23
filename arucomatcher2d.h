#ifndef ARUCOMATCHER2D_H
#define ARUCOMATCHER2D_H

#include "sim_2d.h"
#include "sim_2d_types.h"
#include "cContourBuilder.h"

#include <opencv2/video/video.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp> // /usr/local/include/opencv4/
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>

#include <QObject>

typedef float IMGTYPE;

class ArucoMatcher2D : public SIM_2D
{
    Q_OBJECT
public:
    explicit ArucoMatcher2D(QObject * parent = nullptr);
    virtual ~ArucoMatcher2D();

    // To be possible to implement it with QThread.
    virtual void run();

public slots:
    void setFrame(cv::Mat frame) { m_currentFrame = frame; }

signals:
    void poseEstimated(QVector3D);
    void quat_raw(cv::Mat);

private:
    void init_contours();
    // x, y, phi relative to image SC.
    virtual QVector3D estimate_pose(cv::Mat frame);
    void clearMemory();

private:
    static const int def_dict = cv::aruco::DICT_7X7_50;
    cv::Ptr<cv::aruco::Dictionary> dictionary;
    Vector<int> m_ids_vec;
    UINT m_template_size;

    constexpr static float Marker_size = 0.1f; // [m]
    IMGTYPE *imgGPU, *imgGPU_grad; //,*imgGPU_neg;
    QVector3D m_lastPose;
    cv::Mat m_currentFrame;
    cContoursBuilderGPU m_cbg;
};


#endif // ARUCOMATCHER2D_H
