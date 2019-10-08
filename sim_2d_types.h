#pragma once

#include <ctype.h>
#include <cstdint>

typedef uint16_t MAPTYPE;
typedef uint16_t UINT;
typedef float REAL;
#define Vector std::vector

static const UINT templates_size=256;
static const float PITCH_STEP=0.1f; // [degree]
static const float ROLL_STEP=0.1f; // [degree]
static const float YAW_STEP=0.1f; // [degree]
