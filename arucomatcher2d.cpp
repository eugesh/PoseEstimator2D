#include "arucomatcher2d.h"
#include "cv_math.hpp"


/*
 * SVEN web camera parameters.
 */
static const cv::Mat intrinsic_matrix = (cv::Mat_<double>(3,3) << 1.1327961595847169e+03, 0.0000000000000000e+00, 3.2525414402158128e+02,
                                                                  0.0000000000000000e+00, 1.1475162233789308e+03, 1.3878903347777373e+02,
                                                                  0.0000000000000000e+00, 0.0000000000000000e+00, 1.0000000000000000e+00);

static const cv::Mat distortion_coeff = (cv::Mat_<double>(1,5) <<
              -2.3347424588689411e-01, 1.7924637678196107e+00, -1.8478504713094826e-02, -3.5034777946027344e-03, -6.9593099179000903e+00);

ArucoMatcher2D::ArucoMatcher2D(QObject *parent) : SIM_2D (parent)
{
    dictionary = cv::aruco::getPredefinedDictionary(def_dict);
}

ArucoMatcher2D::~ArucoMatcher2D() {

}

void
ArucoMatcher2D::init_contours() {
    // Create set of contours for the found marker.
}

// x, y, phi relative to image SC.
QVector3D
ArucoMatcher2D::estimate_pose(cv::Mat frame) {
    QVector3D result;

    // Find Aruco marker on image;
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    std::vector<cv::Vec3d> rvecs, tvecs;
    cv::aruco::detectMarkers(frame, dictionary, corners, ids);
    cv::aruco::estimatePoseSingleMarkers(corners, Marker_size, intrinsic_matrix, distortion_coeff, rvecs, tvecs);

    cv::Mat rot_matrix;
    cv::Rodrigues(rvecs, rot_matrix);
    cv::Mat quat = mRot2Quat(rot_matrix);
    emit quat_raw(quat);
    // Gradient of the frame.
    // cv::Mat frame_grad = grad(frame); // ToDO
    // Get set of contours for the found marker (near to found angles).


    // Match contours.

    // Find maximum.

    return result;
}

// To be possible to implement it with QThread.
void
ArucoMatcher2D::run() {
    cv::VideoCapture inputVideo;
    inputVideo.open(0);

    while (inputVideo.grab()) {
        cv::Mat image, imageCopy, EulerAngles;
        std::vector<cv::Vec3d> rvecs, tvecs;

        inputVideo.retrieve(image);
        image.copyTo(imageCopy);
    }

}
