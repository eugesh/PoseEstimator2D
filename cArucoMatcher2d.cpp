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
// #include <windows.h>

const static float Marker_size = 0.13f; // [m]

void
draw_markers(cv::Mat image, std::vector<int> ids, std::vector<cv::Vec3d> rvecs, std::vector<cv::Vec3d> tvecs, std::vector<std::vector<cv::Point2f>> corners) {
    for (size_t i=0; i < ids.size(); i++) {
        cv::aruco::drawDetectedMarkers(image, corners);
        cv::aruco::drawAxis(image, intrinsic_matrix, distortion_coeff, rvecs[i], tvecs[i], 0.1f);
    }
}

void print_contours(cv::Ptr<cv::aruco::Dictionary> dictionary, int template_size, QString name) {
    cv::Mat marker;

    for (int i=0; i < 50; ++i) {
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

    if (DEBUG)
        print_contours(dictionary, m_template_size, "5x5");

    // Convert cv::Mat to QImage
    QImage q_marker_in, q_marker_in_cut, q_marker_tr;
    q_marker_in = ocv::qt::mat_to_qimage_cpy(marker, false);

    // ApplyTransform requires coloured image.
    q_marker_in = q_marker_in.convertToFormat(QImage::Format::Format_ARGB32_Premultiplied).mirrored();

    // Cut template close to "bit" part, no black part.
    QRect boundRect = boundingRect(q_marker_in).marginsAdded(QMargins(1,1,1,1));
    q_marker_in_cut = q_marker_in.copy(boundRect);

    m_cbg.clearAll();
    m_cbg.initAll();

    // Transform Aruco marker: rotate +-5 degree in all dimensions.
    for(float pitch = -PITCH_MAX; pitch < PITCH_MAX; pitch += PITCH_STEP) {
        for(float roll = -ROLL_MAX; roll < ROLL_MAX; roll += ROLL_STEP) {
            for(float yaw = - YAW_MAX; yaw < YAW_MAX; yaw += YAW_STEP) {
                q_marker_tr = ApplyTransform(q_marker_in_cut, QVector3D(0,0,0), QVector3D(pitch, roll, yaw));
                m_cbg.append(q_marker_tr);

                if(DEBUG)
                    q_marker_tr.save(QString("./tr/tr_%1_%2_%3.png").arg(int(pitch*10)).arg(int(roll*10)).arg(int(yaw*10)));
            }
        }
    }

    m_accMatcher.setContoursBuilder(m_cbg);
}

bool
ArucoMatcher2D::estimate_pose(std::vector<cv::Vec3d> & rvecs, std::vector<cv::Vec3d> & tvecs, cv::Mat frame) {

    return true;
}

// x, y, phi relative to image SC.
// Rough estimation by standard library (Aruco lib).
bool
ArucoMatcher2D::estimate_pose(std::vector<int> & ids, std::vector<std::vector<cv::Point2f> > & corners, std::vector<cv::Vec3d> & rvecs, std::vector<cv::Vec3d> & tvecs, cv::Mat shot) {
    // Find Aruco marker on image;

    cv::aruco::detectMarkers(shot, dictionary, corners, ids);
    if (ids.empty())
        return false;

    cv::aruco::estimatePoseSingleMarkers(corners, Marker_size, intrinsic_matrix, distortion_coeff, rvecs, tvecs);

    // Draw markers.
    if (DRAW)
        draw_markers(shot, ids, rvecs, tvecs, corners);
    // for(size_t i=0; i < ids.size() && DRAW; i++)
       // cv::aruco::drawAxis(shot, intrinsic_matrix, distortion_coeff, rvecs[i], tvecs[i], 0.1f);

    return true;
}

/**
 * @brief ArucoMatcher2D::rectify_corners
 * Transform markers coordinates by rvec,tvec transformation.
 * @param corners
 * @param rvec
 * @param tvec
 * @param shot
 * @return
 */
std::vector<cv::Point2f>
ArucoMatcher2D::rectify_corners(std::vector<cv::Point2f> corners, cv::Vec3d const& rvec, cv::Vec3d const& tvec, QImage const& shot, QImage const& shot_new) {
    std::vector<cv::Point2f> new_corners, corners_c = corners, corners_c_m;

    // Center of image;
    QPoint C_shot_pix = QPoint(shot.width() / 2, shot.height() / 2);
    cv::Point2f C_shot_pix_cv = cv::Point2f(shot.width() / 2, shot.height() / 2);

    QPoint C_shot_new_pix = QPoint(shot_new.width() / 2, shot_new.height() / 2);
    cv::Point2f C_shot_new_pix_cv = cv::Point2f(shot_new.width() / 2, shot_new.height() / 2);

    // Find center of marker.
    QPoint C_marker_pix = center_by_diagonals_intersection(corners);
    cv::Point2f C_marker_pix_cv = center_by_diagonals_intersection_cv(corners);

    // Distance between marker and shot centers in pixels
    double pix_dist = std::hypot(C_shot_pix_cv.x - C_marker_pix_cv.x, C_shot_pix_cv.y - C_marker_pix_cv.y);

    // Distance in meters
    double dist_meters = std::hypot(tvec[0], tvec[1]);

    // Find out pixel size in meters (marker_size)
    double dpm = pix_dist / dist_meters; // Dots per meter.

    // Move marker's pixel coordinates to the center.
    for(int i=0; i < corners.size(); ++i)
        corners_c[i] -= C_shot_pix_cv;

    // Convert pixel coordinates to meters.
    for(int i=0; i < corners.size(); ++i)
        corners_c_m.push_back(corners_c[i] / dpm);

    // Transform marker's coordinates to planar image position -> new_corners.
    cv::Vec3d vec0={0,0,0};
    for(int i=0; i < corners_c_m.size(); ++i) {
        new_corners.push_back( qt_math::ApplyTransform_cv(corners_c_m[i], vec0, rvec) );
    }

    // Convert meters to pixel coordinates.
    for(int i=0; i < corners.size(); ++i)
        new_corners[i] = new_corners[i] * dpm;

    // Move marker's pixel coordinates back shot's SC.
    for(int i=0; i < corners.size(); ++i)
        new_corners[i] += C_shot_new_pix_cv;

    return new_corners;
}

ImgArray<IMGTYPE>
ArucoMatcher2D::prepareShot2Matcher(std::vector<cv::Point2f> corners, cv::Vec3d const& rvec, cv::Vec3d const& tvec, QImage const& shot) {
    QImage qimg_planar, cut_shot, cut_shot_scaled;

    // Cut image by corners.
    // QRect cornersRect = rectangleFromCorners(corners).marginsAdded(QMargins(ROI_MARGIN, ROI_MARGIN, ROI_MARGIN, ROI_MARGIN));
    // cut_shot = shot.copy(cornersRect);

    // Estimate


    // Convert Rodrigues to Quaterninon.
    QQuaternion quat = rvec2QQaternion(rvec); //.inverted();

    // Convert quaternion to Euler.
    QVector3D EulerAngles = quat.toEulerAngles();

    // Apply affine transform.
    // Transform image: rotate with estimated by Aruco lib quaternion.
    QVector3D Tr = QVector3D(tvec[0], tvec[1], tvec[2]);
    qimg_planar = ApplyTransform(shot, QVector3D(0,0,0), EulerAngles);

    // Transform corners' coordinates.
    // First, translate frame to the center of SC.
    std::vector<cv::Point2f> new_corners = rectify_corners(corners, rvec, tvec, shot, qimg_planar);

    // Cut shot inside corners.
    QRect bound_rect = rectangleFromCorners(new_corners);
    cut_shot = qimg_planar.copy(bound_rect);

    // Scale block to fit template size + margin.   ToDo: generate a range of scaled images.
    if(cut_shot.width() < 1)
        return ImgArray<IMGTYPE>();

    double scale = double(templates_size + 2*ROI_MARGIN) / double(cut_shot.width());
    cut_shot_scaled = cut_shot.scaled(cut_shot.size() * scale);

    // Apply Sobel mask.
    // Apply sobel filter -> gradient.
    ImgArray<IMGTYPE> img_ar(cut_shot_scaled);
    m_img_arr_grad.setImage(cut_shot_scaled);
    create_matr_gradXY(m_img_arr_grad.getArray(), img_ar.width(), img_ar.height(), img_ar.getArray());

    if (DRAW) {
        // Show transformed image.
        cv::Mat img_tmp = ocv::qt::qimage_to_mat_cpy(qimg_planar, false);
        cv::Mat img_tmp_copy;
                img_tmp.copyTo(img_tmp_copy);
        // cv::aruco::drawDetectedMarkers(img_tmp_copy, new_corners);
        cv::line(img_tmp, new_corners[0], new_corners[1], cv::Scalar( 255 ), 3);
        cv::line(img_tmp, new_corners[1], new_corners[2], cv::Scalar( 255 ), 3);
        cv::line(img_tmp, new_corners[2], new_corners[3], cv::Scalar( 255 ), 3);
        cv::line(img_tmp, new_corners[3], new_corners[0], cv::Scalar( 255 ), 3);
        cv::imshow("frame_planar", img_tmp);
    }

    if (DEBUG && ! m_img_arr_grad.empty()) {
        // Save image.
        float range = (m_img_arr_grad.max() - m_img_arr_grad.min());
        m_img_arr_grad = m_img_arr_grad * float(255) / range;
        range = (img_ar.max() - img_ar.min());
        img_ar.toQImage().save("Before_grad.png");
        m_img_arr_grad.toQImage().save("GRADIENT.png");
        qimg_planar.save("planar.png");
        cut_shot.save("Cut_shot.png");
        cut_shot_scaled.save("cut_shot_scaled.png");
    }

    return m_img_arr_grad;
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

    if (inputVideo.open(0)) {
        while (inputVideo.grab()) {
            cv::Mat image, imageCopy, EulerAngles;
            std::vector<cv::Vec3d> rvecs, tvecs;
            std::vector<int> ids;
            std::vector<std::vector<cv::Point2f> > corners;
            QImage qImageCopy, qImg_planar_grad;

            inputVideo.retrieve(image);
            image.copyTo(imageCopy);
            qImageCopy = ocv::qt::mat_to_qimage_cpy(image, false);

            // Estimate pose with Aruco lib.
            if (estimate_pose(ids, corners, rvecs, tvecs, imageCopy)) {

                // Apply Transfrom
                prepareShot2Matcher(corners.front(), rvecs.front(), tvecs.front(), qImageCopy);

                // Run Accurate matcher -> estimate delta rotation shift.
                // estimate_poseAccurate
                m_accMatcher.estimate(rvecs.front(), tvecs.front(), m_img_arr_grad);
            }

            // Draw marker: compare Aruco and Accurate matcher results.
            cv::imshow("frame", imageCopy);
            char key = char(cv::waitKey(3));
            if(key == 27)
                return;
        }
    }
    else if (DEBUG){
        // Stub: Infinite loop with the same image;
        // QImage img2test("D:/workspace/projects/PoseEstimator2D/img2test.png");
        // img2test.save("D:/workspace/projects/PoseEstimator2D/img2test2.png");

        forever {
            cv::Mat image, imageCopy, EulerAngles;
            std::vector<cv::Vec3d> rvecs, tvecs;
            std::vector<int> ids;
            std::vector<std::vector<cv::Point2f> > corners;
            QImage qImageCopy, qImg_planar_grad;

            // inputVideo.retrieve(image);
            // qImageCopy = img2test; //ocv::qt::mat_to_qimage_cpy(image, false);
            // image = ocv::qt::qimage_to_mat_cpy(img2test);
            image = cv::imread("D:/workspace/projects/PoseEstimator2D/planar4.png");// , CV_LOAD_IMAGE_COLOR);
            // imageCopy = cv::imread("D:/workspace/projects/PoseEstimator2D/planar.png");// , CV_LOAD_IMAGE_COLOR);
            image.copyTo(imageCopy);
            qImageCopy = ocv::qt::mat_to_qimage_cpy(image, false);
            // cv::imwrite("D:/workspace/projects/PoseEstimator2D/img2test3.png", imageCopy);

            // Estimate pose with Aruco lib.
            if (estimate_pose(ids, corners, rvecs, tvecs, imageCopy)) {

                // Apply Transfrom
                prepareShot2Matcher(corners.front(), rvecs.front(), tvecs.front(), qImageCopy);

                // Run Accurate matcher -> estimate delta rotation shift.
                // estimate_poseAccurate
                m_accMatcher.estimate(rvecs.front(), tvecs.front(), m_img_arr_grad);
            }

            // Draw marker: compare Aruco and Accurate matcher results.
            cv::imshow("frame", imageCopy);
            char key = char(cv::waitKey(3));
            if(key == 27)
                return;

             // Sleep(10000);
        }
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
