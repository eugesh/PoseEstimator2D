#pragma once

#include <QVector3D>
#include <QTransform>
#include <QMatrix4x4>
#include <QImage>

QTransform
getAffineTransform (float x, float y, float z, float phi_x, float phi_y, float phi_z) {
    QMatrix4x4 matr;

    matr.translate(x, y, z);
    matr.rotate(phi_x, 1, 0);
    matr.rotate(phi_y, 0, 1);
    matr.rotate(phi_z, 0, 0, 1);

    return matr.toTransform();
}

QTransform
getAffineTransform (QVector3D R, QVector3D t) {
    return getAffineTransform (R.x(), R.y(), R.z(), t.x(), t.y(), t.z());
}

/**
 * @brief getTransformed
 * @param Apply transform to img itself.
 * @return
 */
/*QImage
getTransformed(QImage & img, QVector3D R, QVector3D t) {
    QTransform tf = getAffineTransform(R, t);

     img.transformed();
}*/

/**
 * @brief ApplyTransform
 * @param Apply transform to img and get copy of it.
 * @return
 */
QImage
ApplyTransform(QImage const& img, QVector3D R, QVector3D t) {
    QImage outImg(img);

    QTransform tf = getAffineTransform(R, t);

    outImg.transformed(tf);

    return outImg;
}
