#include "cArucoMatcher2d.h"
#include "cv_math.hpp"
#include "qt_math.hpp"
#include "cContourBuilder.h"
#include "mat_qimage.hpp"

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
//#include <cufft.h>
#include <cuda.h>
#include <math.h>

void
draw_markers(cv::Mat image, std::vector<int> ids, std::vector<cv::Vec3d> rvecs, std::vector<cv::Vec3d> tvecs, std::vector<std::vector<cv::Point2f>> corners) {
    for(size_t i=0; i < ids.size(); i++) {
        cv::aruco::drawDetectedMarkers(image, corners);
        cv::aruco::drawAxis(image, intrinsic_matrix, distortion_coeff, rvecs[i], tvecs[i], 0.1f);
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

ArucoMatcher2D::ArucoMatcher2D(QObject *parent) : SIM_2D (parent)
{
    dictionary = cv::aruco::getPredefinedDictionary(def_dict);
    m_ids_vec.push_back(2);
    m_template_size = templates_size;
    init_contours();
    m_cbg.contoursSetup();
}

ArucoMatcher2D::~ArucoMatcher2D() {

}

void
ArucoMatcher2D::init_contours() {
    // Create set of contours for the found marker.

    // Get Aruco marker.
    // CV_EXPORTS_W void drawMarker(const Ptr<Dictionary> &dictionary, int id, int sidePixels, OutputArray img, int borderBits = 1);
    // CV_WRAP_AS(create) static Ptr<Dictionary> create(int nMarkers, int markerSize, int randomSeed=0);
    // dictionary->create(50, m_template_size);
    cv::Mat marker;
    cv::aruco::drawMarker(dictionary, m_ids_vec.front(), m_template_size, marker);

    if(DEBUG)
        print_contours(dictionary, m_template_size, "5x5");

    // Convert cv::Mat to QImage
    QImage q_marker_in, q_marker_tr;
    q_marker_in = ocv::qt::mat_to_qimage_cpy(marker, false);

    q_marker_in = q_marker_in.convertToFormat(QImage::Format::Format_ARGB32_Premultiplied).mirrored();

    m_cbg.clearAll();
    m_cbg.initAll();

    // Transform Aruco marker: rotate +-5 degree in all dimensions.
    for(float pitch = -PITCH_MAX; pitch < PITCH_MAX; pitch += PITCH_STEP) {
        for(float roll = -ROLL_MAX; roll < ROLL_MAX; roll += ROLL_STEP) {
            for(float yaw = - YAW_MAX; yaw < YAW_MAX; yaw += YAW_STEP) {
                q_marker_tr = ApplyTransform(q_marker_in, QVector3D(0,0,0), QVector3D(pitch, roll, yaw));
                m_cbg.append(q_marker_tr);

                if(DEBUG)
                    q_marker_tr.save(QString("./tr/tr_%1_%2_%3.png").arg(int(pitch*10)).arg(int(roll*10)).arg(int(yaw*10)));
            }
        }
    }
}

// x, y, phi relative to image SC.
// Rough estimation by standard library (Aruco lib).
bool
ArucoMatcher2D::estimate_pose(std::vector<cv::Vec3d> & rvecs, std::vector<cv::Vec3d> & tvecs, cv::Mat shot) {
    // Find Aruco marker on image;
    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    cv::aruco::detectMarkers(shot, dictionary, corners, ids);
    if(ids.empty())
        return false;

    cv::aruco::estimatePoseSingleMarkers(corners, Marker_size, intrinsic_matrix, distortion_coeff, rvecs, tvecs);

    // Draw markers.
    if(DRAW)
        draw_markers(shot, ids, rvecs, tvecs, corners);
    // for(size_t i=0; i < ids.size() && DRAW; i++)
       // cv::aruco::drawAxis(shot, intrinsic_matrix, distortion_coeff, rvecs[i], tvecs[i], 0.1f);

    return true;
}

ImgArray<float>
ArucoMatcher2D::prepareShot2Matcher(cv::Vec3d const& rvec, cv::Vec3d const& tvec, QImage const& shot) {
    QImage qimg_planar;

    // Convert Rodrigues to Quaterninon.
    QQuaternion quat = rvec2QQaternion(rvec);

    // Convert quaternion to Euler.
    QVector3D EulerAngles = quat.toEulerAngles();

    // Apply affine transform.
    // Transform image: rotate with estimated by Aruco lib quaternion.
    QVector3D Tr = QVector3D(tvec[0], tvec[1], tvec[2]);
    qimg_planar = ApplyTransform(shot, Tr, EulerAngles);

    // Apply Sobel mask.
    // Apply sobel filter -> gradient.
    ImgArray<float> img_ar(qimg_planar), img_ar_grad(qimg_planar);
    create_matr_gradXY(img_ar_grad.getArray(), img_ar.width(), img_ar.height(), img_ar.getArray());

    if(DEBUG) {
        // Save image.
        float range = (img_ar_grad.max() - img_ar_grad.min());
        img_ar_grad = img_ar_grad * float(255) / range;
        range = (img_ar.max() - img_ar.min());
        img_ar.toQImage().save("Before_grad.png");
        img_ar_grad.toQImage().save("GRADIENT.png");
        qimg_planar.save("planar.png");
    }

    return img_ar_grad;
}

bool
ArucoMatcher2D::estimate_poseAccurate(std::vector<cv::Vec3d> & rvecs, std::vector<cv::Vec3d> & tvecs, cv::Mat shot) {
    // Get (generate) set of contours for the found marker (near to found angles).


    // Match contours.
    // Estimate mean values for each seeking contour -> build table of mean values.


    // Find maximum in the table.


    return true;
}

// To be possible to implement it with QThread.
void
ArucoMatcher2D::run() {
    cv::VideoCapture inputVideo;
    inputVideo.open(0);

    while (inputVideo.grab()) {
        cv::Mat image, imageCopy, EulerAngles;
        std::vector<cv::Vec3d> rvecs, tvecs;
        QImage qImageCopy, qImg_planar_grad;

        inputVideo.retrieve(image);
        image.copyTo(imageCopy);
        qImageCopy = ocv::qt::mat_to_qimage_cpy(image, false);

        // Estimate pose with Aruco lib.
        if(estimate_pose(rvecs, tvecs, imageCopy)) {
            ImgArray<float> img_arr = prepareShot2Matcher(rvecs.front(), tvecs.front(), qImageCopy);

            // Run Accurate matcher -> estimate delta rotation shift.
            // estimate_poseAccurate
        }

        // Draw marker: compare Aruco and Accurate matcher results.
        cv::imshow("frame", imageCopy);
        char key = char(cv::waitKey(3));
        if(key == 27)
            return;
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

/*cv::Mat
ArucoMatcher2D::prepareShot2Matcher(std::vector<cv::Vec3d> & rvecs, std::vector<cv::Vec3d> & tvecs, cv::Mat shot) {
    cv::Mat img_planar_grad;

    // Apply affine transform.


    // Apply Sobel mask.


    return img_planar_grad;
}*/

/*QImage
ArucoMatcher2D::prepareShot2Matcher(cv::Vec3d const& rvec, cv::Vec3d const& tvec, QImage const& shot) {
    QImage qimg_planar;

    // Apply affine transform.

    // Transform image: rotate with estimated by Aruco lib quaternion.
    QVector3D rough_pose3D (rvec[0], rvec[1], rvec[2]);
    qimg_planar = ApplyTransform(shot, QVector3D(0,0,0), rough_pose3D);

    // Apply Sobel mask.
    // Apply sobel filter -> gradient.
    ImgArray<float> img_ar(qimg_planar), img_ar_grad(qimg_planar);
    create_matr_gradXY(img_ar_grad.getArray(), img_ar.width(), img_ar.height(), img_ar.getArray());

    return img_ar_grad.toQImage();
}*/

/*ImgArray<float>
ArucoMatcher2D::prepareShot2Matcher(cv::Vec3d const& rvec, cv::Vec3d const& tvec, QImage const& shot) {
    QImage qimg_planar;

    // Convert Rodrigues to Quaterninon.
    QQuaternion quat = rvec2QQaternion(rvec);

    // Convert quaternion to Euler.
    cv::Mat EulerAngles = rvec2Euler(rvec);
    quat.toEulerAngles();

    // Apply affine transform.
    // Transform image: rotate with estimated by Aruco lib quaternion.
    QVector3D rough_pose3D = QVector3D(EulerAngles.at<double>(0), EulerAngles.at<double>(1), EulerAngles.at<double>(2));
    rough_pose3D *= float(180.0 / M_PI);
    QVector3D Tr = QVector3D(tvec[0], tvec[1], tvec[2]);
    qimg_planar = ApplyTransform(shot, Tr, rough_pose3D);

    // Apply Sobel mask.
    // Apply sobel filter -> gradient.
    ImgArray<float> img_ar(qimg_planar), img_ar_grad(qimg_planar);
    create_matr_gradXY(img_ar_grad.getArray(), img_ar.width(), img_ar.height(), img_ar.getArray());

    if(DEBUG) {
        // Save image.
        float range = (img_ar_grad.max() - img_ar_grad.min());
        img_ar_grad = img_ar_grad * float(255) / range;
        range = (img_ar.max() - img_ar.min());
        img_ar.toQImage().save("Before_grad.png");
        img_ar_grad.toQImage().save("GRADIENT.png");
        qimg_planar.save("planar.png");
    }

    return img_ar_grad;
}*/
