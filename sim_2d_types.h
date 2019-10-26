#pragma once

#include <ctype.h>
#include <cstdint>
#include <QPoint>


typedef int MAPTYPE;
typedef int UINT;
typedef float REAL;
#define Vector std::vector
typedef std::vector<QPoint> Contour;
typedef std::vector<UINT> SparseContour;

static const UINT templates_size=32;
static const float PITCH_STEP=0.5f; // [degree]
static const float ROLL_STEP=0.5f; // [degree]
static const float YAW_STEP=0.5f; // [degree]

static const float PITCH_MAX=5.0f; // [degree]
static const float ROLL_MAX=5.0f; // [degree]
static const float YAW_MAX=5.0f; // [degree]

static const bool debug=true;
