#pragma once

#include <ctype.h>
#include <cstdint>
#include <QPoint>
#include <opencv2/aruco.hpp>

typedef float IMGTYPE;
typedef int MAPTYPE;
typedef int INT;
typedef float REAL;
#define Vector std::vector
static const float EPS_FLOAT=0.000001f;
typedef std::vector<QPoint> Contour;
typedef std::vector<INT> SparseContour;

static const INT templates_size=32;
static const INT ROI_MARGIN=16;
static const float PITCH_STEP=0.5f; // [degree]
static const float ROLL_STEP=0.5f; // [degree]
static const float YAW_STEP=0.5f; // [degree]

static const float PITCH_MAX=5.0f; // [degree]
static const float ROLL_MAX=5.0f; // [degree]
static const float YAW_MAX=5.0f; // [degree]

static const bool SHOW=true;
static const bool DEBUG=false;
static const bool DRAW=true;
static const int def_dict = cv::aruco::DICT_5X5_50;
