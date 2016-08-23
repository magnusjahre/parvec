// SIMD Version by Juan M. Cebrian, NTNU - 2013. (modifications under JMCG tag)

#ifndef SIMD_HEADER_H
#define SIMD_HEADER_H

/* JMCG BEGIN */
#ifdef PARSEC_USE_SSE
#include "simd_defines.h"
#endif

#ifdef PARSEC_USE_AVX
#include "simd_defines.h"
#endif

#ifdef PARSEC_USE_AVX512
#include "simd_defines.h"
#endif

#ifdef PARSEC_USE_NEON
#include "simd_defines.h"
#endif
/* JMCG END */

//#define DEBUG_SIMD

#endif // SIMD_HEADER_H
