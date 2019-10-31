#ifndef ARUCOMATCHER2D_H
#define ARUCOMATCHER2D_H

#include "sim_2d.h"
#include "sim_2d_types.h"
#include "cContourBuilder.h"
#include "camera_param.hpp"
#include "cAccurateMatcherGPU.h"

// #include <opencv2/video/video.hpp>
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
    bool estimate_pose(std::vector<cv::Vec3d> & rvecs, std::vector<cv::Vec3d> & tvecs, cv::Mat frame);

public slots:
    void setFrame(cv::Mat frame) { m_currentFrame = frame; }

signals:
    void poseEstimated(QVector3D);
    void quat_raw(cv::Mat);

private:
    void init_contours();
    // x, y, phi relative to image SC.
    bool estimate_poseAccurate(std::vector<cv::Vec3d> & rvecs, std::vector<cv::Vec3d> & tvecs, cv::Mat shot);
    // cv::Mat prepareShot2Matcher(std::vector<cv::Vec3d> & rvecs, std::vector<cv::Vec3d> & tvecs, cv::Mat frame); // Implement
    QImage prepareShot2Matcher(cv::Vec3d const& rvec, cv::Vec3d const& tvec, QImage const& shot);
    void clearMemory();

private:
    cv::Ptr<cv::aruco::Dictionary> dictionary;
    Vector<int> m_ids_vec;
    INT m_template_size;

    constexpr static float Marker_size = 0.1f; // [m]
    IMGTYPE *imgGPU, *imgGPU_grad; //,*imgGPU_neg;
    QVector3D m_lastPose;
    cv::Mat m_currentFrame;
    cContoursBuilderGPU m_cbg;
    cAccurateMatcherGPU m_accMatcher;
};


#endif // ARUCOMATCHER2D_H
