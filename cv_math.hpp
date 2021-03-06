#ifndef CV_MATH_HPP
#define CV_MATH_HPP

#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/opencv.hpp>
#include <QQuaternion>
#include <QRect>
#include <QPolygon>


inline float SIGN(float x) {
    return (x >= 0.0f) ? +1.0f : -1.0f;
}

inline float NORM(float a, float b, float c, float d) {
    return sqrt(a * a + b * b + c * c + d * d);
}

// quaternion = [w, x, y, z]'
cv::Mat mRot2Quat(const cv::Mat& m) {
    float r11 = m.at<float>(0, 0);
    float r12 = m.at<float>(0, 1);
    float r13 = m.at<float>(0, 2);
    float r21 = m.at<float>(1, 0);
    float r22 = m.at<float>(1, 1);
    float r23 = m.at<float>(1, 2);
    float r31 = m.at<float>(2, 0);
    float r32 = m.at<float>(2, 1);
    float r33 = m.at<float>(2, 2);
    float q0 = (r11 + r22 + r33 + 1.0f) / 4.0f;
    float q1 = (r11 - r22 - r33 + 1.0f) / 4.0f;
    float q2 = (-r11 + r22 - r33 + 1.0f) / 4.0f;
    float q3 = (-r11 - r22 + r33 + 1.0f) / 4.0f;
    if (q0 < 0.0f) {
        q0 = 0.0f;
    }
    if (q1 < 0.0f) {
        q1 = 0.0f;
    }
    if (q2 < 0.0f) {
        q2 = 0.0f;
    }
    if (q3 < 0.0f) {
        q3 = 0.0f;
    }
    q0 = sqrt(q0);
    q1 = sqrt(q1);
    q2 = sqrt(q2);
    q3 = sqrt(q3);
    if (q0 >= q1 && q0 >= q2 && q0 >= q3) {
        q0 *= +1.0f;
        q1 *= SIGN(r32 - r23);
        q2 *= SIGN(r13 - r31);
        q3 *= SIGN(r21 - r12);
    }
    else if (q1 >= q0 && q1 >= q2 && q1 >= q3) {
        q0 *= SIGN(r32 - r23);
        q1 *= +1.0f;
        q2 *= SIGN(r21 + r12);
        q3 *= SIGN(r13 + r31);
    }
    else if (q2 >= q0 && q2 >= q1 && q2 >= q3) {
        q0 *= SIGN(r13 - r31);
        q1 *= SIGN(r21 + r12);
        q2 *= +1.0f;
        q3 *= SIGN(r32 + r23);
    }
    else if (q3 >= q0 && q3 >= q1 && q3 >= q2) {
        q0 *= SIGN(r21 - r12);
        q1 *= SIGN(r31 + r13);
        q2 *= SIGN(r32 + r23);
        q3 *= +1.0f;
    }
    else {
        printf("coding error\n");
    }
    float r = NORM(q0, q1, q2, q3);
    q0 /= r;
    q1 /= r;
    q2 /= r;
    q3 /= r;

    cv::Mat res = (cv::Mat_<float>(4, 1) << q0, q1, q2, q3);
    return res;
}

/* Calculates rotation matrix to euler angles
   The result is the same as MATLAB except the order
   of the euler angles ( x and z are swapped ). */
/*cv::Mat rotationMatrixToEulerAngles(R) {

    assert(isRotationMatrix(R))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z])
} */

cv::Mat rotMatrix2Euler(const cv::Mat & rotationMatrix)
{
    cv::Mat euler(3, 1, CV_64F);

    double m00 = rotationMatrix.at<double>(0, 0);
    double m02 = rotationMatrix.at<double>(0, 2);
    double m10 = rotationMatrix.at<double>(1, 0);
    double m11 = rotationMatrix.at<double>(1, 1);
    double m12 = rotationMatrix.at<double>(1, 2);
    double m20 = rotationMatrix.at<double>(2, 0);
    double m22 = rotationMatrix.at<double>(2, 2);

    double x, y, z;

    // Assuming the angles are in radians.
    if (m10 > 0.998) { // singularity at north pole
        x = 0;
        y = CV_PI / 2;
        z = atan2(m02, m22);
    }
    else if (m10 < -0.998) { // singularity at south pole
        x = 0;
        y = -CV_PI / 2;
        z = atan2(m02, m22);
    }
    else {
        x = atan2(-m12, m11);
        y = asin(m10);
        z = atan2(-m20, m00);
    }

    euler.at<double>(0) = x;
    euler.at<double>(1) = y;
    euler.at<double>(2) = z;

    return euler;
}

cv::Mat
rvec2Euler (cv::Vec3d rvec) {
    cv::Mat EulerAngles;
    cv::Mat rot_matrix;

    cv::Rodrigues(rvec, rot_matrix);
    // cv::Mat quat = mRot2Quat(rot_matrix);
    EulerAngles = rotMatrix2Euler(rot_matrix);

    return EulerAngles;
}

cv::Mat
rvec2Quaternion (cv::Vec3d rvec) {
    cv::Mat EulerAngles;
    cv::Mat rot_matrix;

    cv::Rodrigues(rvec, rot_matrix);
    cv::Mat quat = mRot2Quat(rot_matrix);
    // EulerAngles = rotMatrix2Euler(rot_matrix);

    return quat;
}

QQuaternion
rvec2QQaternion (cv::Vec3d rvec) {
    QQuaternion quat;

    double xr = rvec[0];
    double yr = rvec[1];
    double zr = rvec[2];

    float theta = (float)(sqrt(xr*xr + yr*yr + zr*zr) * 180 / CV_PI);
    QVector3D axis = QVector3D(xr, yr, zr);
    // Quaternion rot = Quaternion.AngleAxis (theta, axis);

    quat = QQuaternion::fromAxisAndAngle(axis, theta);

    return quat;
}
// QQuaternion QQuaternion::fromAxisAndAngle(const QVector3D &axis, float angle)
// Creates a normalized quaternion that corresponds to rotating through angle degrees about the specified 3D axis.


QRect
rectangleFromCorners(std::vector<cv::Point2f> corners) {
    QPoint leftTop, bottomRight;
    int leftMin=std::numeric_limits<int>::max(), topMin=std::numeric_limits<int>::max(), rightMax=0, bottomMax=0;

    for(int i=0; i < corners.size(); ++i) {
        if(corners.at(i).x < leftMin) {
            leftMin = corners.at(i).x;
        }
        if(corners.at(i).x > rightMax) {
            rightMax = corners.at(i).x;
        }
        if(corners.at(i).y > bottomMax) {
            bottomMax = corners.at(i).y;
        }
        if(corners.at(i).y < topMin) {
            topMin = corners.at(i).y;
        }
    }

    return QRect(QPoint(leftMin, topMin), QPoint(rightMax, bottomMax));
}

QPolygon polygonConverter (std::vector<cv::Point2f> corners) {
    QVector<QPoint> points;

    for(int i=0; i < corners.size(); ++i) {
        points.push_back(QPoint(corners.at(i).x, corners.at(i).y));
    }

    return QPolygon(points);
}

QPoint center_by_diagonals_intersection(std::vector<cv::Point2f> corners) {
    auto x1 = corners.at(0).x;
    auto y1 = corners.at(0).y;
    auto x3 = corners.at(1).x;
    auto y3 = corners.at(1).y;
    auto x2 = corners.at(2).x;
    auto y2 = corners.at(2).y;
    auto x4 = corners.at(3).x;
    auto y4 = corners.at(3).y;
    auto Cx = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) /
         ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4));
    auto Cy = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) /
         ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4));

    return QPoint(Cx, Cy);
}

cv::Point2f center_by_diagonals_intersection_cv(std::vector<cv::Point2f> corners) {
    auto x1 = corners.at(0).x;
    auto y1 = corners.at(0).y;
    auto x3 = corners.at(1).x;
    auto y3 = corners.at(1).y;
    auto x2 = corners.at(2).x;
    auto y2 = corners.at(2).y;
    auto x4 = corners.at(3).x;
    auto y4 = corners.at(3).y;
    auto Cx = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) /
         ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4));
    auto Cy = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) /
         ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4));

    return cv::Point2f(Cx, Cy);
}

float
distance_from_points(cv::Point2f p1, cv::Point2f p2) {
    float dist=0;



    return dist;
}

/**
 * @brief estimate_dpm_for_marker
 * Rougly estimates dots(pixels) per meter for Aruco marker.
 * @param corners
 * @param marker_size
 * @return
 */
float
estimate_dpm_for_marker(std::vector<cv::Point2f> corners, float marker_size) {
    float dpm = 1.0f;

    return dpm;
}

#ifdef QT_CORE_LIB
#include "qt_math.hpp"

namespace qt_math {

QVector3D rvec2Euler(cv::Vec3d const& rvec) {
    // Convert Rodrigues to Quaterninon.
    QQuaternion quat = rvec2QQaternion(rvec);

    // Convert quaternion to Euler.
    return quat.toEulerAngles();
}

QPoint
ApplyTransform(cv::Point2f point, QVector3D t, QVector3D R) {
    return ApplyTransform(QPoint(point.x, point.y), t, R);
}

cv::Point2f
ApplyTransform_cv(cv::Point2f point, QVector3D t, QVector3D R) {
    QPoint qpoint_out = ApplyTransform(QPoint(point.x, point.y), t, R);

    cv::Point2f out = cv::Point2f(qpoint_out.x(), qpoint_out.y());

    return out;
}

cv::Point2f
ApplyTransform_cv(cv::Point2f point, cv::Vec3d const& tvec, cv::Vec3d const& rvec) {
    QVector3D R = rvec2Euler(rvec);
    QVector3D t = QVector3D(tvec[0], tvec[1], tvec[2]);

    QPointF qpoint_out = ApplyTransform(QPointF(point.x, point.y), t, R);

    cv::Point2f out = cv::Point2f(qpoint_out.x(), qpoint_out.y());

    return out;
}

std::vector<cv::Point2f> polygon2vector (QPolygon polygon) {
    std::vector<cv::Point2f> vecPoints;

    for(int i=0; i < polygon.count(); ++i) {
        vecPoints.push_back(cv::Point2f(polygon[i].x(), polygon[i].y()));
    }

    return vecPoints;
}

}
#endif

#endif // CV_MATH_HPP
