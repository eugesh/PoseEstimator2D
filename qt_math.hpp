#pragma once

#include <QVector3D>
#include <QTransform>
#include <QMatrix4x4>
#include <QImage>

/**
 * Sobel operator, gradient.
 */
template <typename T>
void create_matr_gradXY(T*out, int W, int H, T*in) {

    for(int i=1; i < H - 1; ++i) {
      for(int j=1; j < W - 1; ++j) {
         out[i * W + j] = fabs(float(-in[(i-1) * W + j - 1] + in[(i-1) * W + j + 1] -
                            2 * in[i * W + j - 1] + 2 * in[i * W + j + 1] - in[(i+1) * W + j - 1] + in[(i+1) * W + j + 1])) +
                         fabs(float(in[(i-1) * W + j - 1] + 2 * in[(i-1) * W + j] + in[(i-1) * W + j + 1] -
                            in[(i+1) * W + j - 1] - 2 * in[(i+1) * W + j] - in[(i+1) * W + j + 1]));
      }
    }
}

QImage sobel(QImage const& img) {
    QImage grad;



    for(int i=0; i < img.height(); ++i) {
        for(int j=0; j < img.width(); ++j) {

        }
    }

    return grad;
}

QMatrix4x4
getAffineMatrix (float x, float y, float z, float phi_x, float phi_y, float phi_z) {
    QMatrix4x4 matr;

    matr.translate(x, y, z);
    matr.rotate(phi_x, 1, 0, 0);
    matr.rotate(phi_y, 0, 1, 0);
    matr.rotate(phi_z, 0, 0, 1);

    return matr;
}

QTransform
getAffineTransform (float x, float y, float z, float phi_x, float phi_y, float phi_z) {
    QMatrix4x4 matr;

    matr.translate(x, y, z);
    matr.rotate(phi_x, 1, 0, 0);
    matr.rotate(phi_y, 0, 1, 0);
    matr.rotate(phi_z, 0, 0, 1);

    return matr.toTransform();
}

QTransform
getAffineTransform (QVector3D t, QVector3D R) {
    return getAffineTransform (t.x(), t.y(), t.z(), R.x(), R.y(), R.z());
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
ApplyTransform(QImage const& img, QVector3D t, QVector3D R) {
    QImage outImg;

    QTransform tf = getAffineTransform(t, R);

    outImg = img.transformed(tf);

    return outImg;
}

QRect
ApplyTransform(QRect const& rect, QVector3D t, QVector3D R) {
    QRect outRect;

    QMatrix4x4 matr = getAffineMatrix (t.x(), t.y(), t.z(), R.x(), R.y(), R.z());

    outRect = matr.mapRect(rect);

    return outRect;
}
