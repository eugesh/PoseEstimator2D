#include "arucomatcher2d.h"
#include "cv_math.hpp"
#include "qt_math.hpp"
#include "cContourBuilder.h"
#include "mat_qimage.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include <cufft.h>
#include <cuda.h>
#include <math.h>

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
    m_ids_vec.push_back(0);
    m_template_size = templates_size;
    init_contours();
}

ArucoMatcher2D::~ArucoMatcher2D() {

}

void
ArucoMatcher2D::init_contours() {
    // Create set of contours for the found marker.
    cContoursBuilderGPU cbg;

    // Get Aruco marker.
    // CV_EXPORTS_W void drawMarker(const Ptr<Dictionary> &dictionary, int id, int sidePixels, OutputArray img, int borderBits = 1);
    // CV_WRAP_AS(create) static Ptr<Dictionary> create(int nMarkers, int markerSize, int randomSeed=0);
    // dictionary->create(50, m_template_size);
    cv::Mat marker;
    cv::aruco::drawMarker(dictionary, m_ids_vec.front(), m_template_size, marker);

    // Convert cv::Mat to QImage
    QImage q_marker_in, q_marker_tr;
    q_marker_in = ocv::qt::mat_to_qimage_cpy(marker, false);

    q_marker_in = q_marker_in.convertToFormat(QImage::Format::Format_ARGB32_Premultiplied);

    // Transform Aruco marker: rotate +-5 degree in all dimensions.
    for(float pitch = -PITCH_MAX; pitch < PITCH_MAX; pitch += PITCH_STEP) {
        for(float roll = -ROLL_MAX; roll < ROLL_MAX; roll += ROLL_STEP) {
            for(float yaw = - YAW_MAX; yaw < YAW_MAX; yaw += YAW_STEP) {
                q_marker_tr = ApplyTransform(q_marker_in, QVector3D(0,0,0), QVector3D(pitch, roll, yaw));
                cbg.append(q_marker_tr);

                if(debug)
                    q_marker_tr.save(QString("./tr/tr_%1_%2_%3.png").arg(int(pitch*10)).arg(int(roll*10)).arg(int(yaw*10)));
            }
        }
    }
}

void print_contours(cv::Ptr<cv::aruco::Dictionary> dictionary, int template_size, QString name) {
    cv::Mat marker;

    for(int i=0; i < 50; ++i) {
        cv::aruco::drawMarker(dictionary, i, template_size, marker);
        if(marker.cols > 0)
            cv::imwrite(QString("%1_id%2.png").arg(name).arg(i).toUtf8().toStdString(), marker);
    }
    // cv::namedWindow ("marker", 1);

    if(marker.cols > 0) {
        cv::imshow("marker", marker);
        char key = char (cv::waitKey(3));
        if (key == 27)
            return;
    }
}

// x, y, phi relative to image SC.
// Rough estimation by standard library (Aruco lib).
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
        QImage qImageCopy, qImageCopy_pl_rough;

        inputVideo.retrieve(image);
        image.copyTo(imageCopy);

        // Estimate pose with Aruco lib.
        QVector3D rough_pose3D = estimate_pose(imageCopy);
        qImageCopy = ocv::qt::mat_to_qimage_cpy(imageCopy, false);

        // Transform image: rotate with estimated by Aruco lib quaternion.
        qImageCopy_pl_rough = ApplyTransform(qImageCopy, QVector3D(0,0,0), rough_pose3D);

        // Apply sobel filter -> gradient.
        // qImageCopy_pl_rough
        // Run Accurate matcher -> estimate delta rotation shift.

        // Draw marker: compare Aruco and Accurate matcher results.

    }

}

void
ArucoMatcher2D::clearMemory() {
    if(imgGPU!=nullptr) {
       cudaFree(imgGPU     );
       imgGPU=nullptr;
    }
    if(imgGPU_grad!=nullptr) {
       cudaFree(imgGPU_grad);
       imgGPU_grad=nullptr;
    }
}
