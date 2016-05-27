// Helper functions made by Hallgeir Lien (hallgeir.lien@gmail.com)

#ifndef __HELPERS_H__
#define __HELPERS_H__


#include <stdio.h>
#if defined(PARSEC_USE_SSE) || defined(PARSEC_USE_AVX)
#include <immintrin.h>
#endif

typedef float  __attribute__((vector_size(16))) vsf4;

inline void printv_arr(float v[4])
{
    printf("%e %e %e %e\n", v[0], v[1], v[2], v[3]);
}

inline void printv_arrx(float v[4])
{
    printf("%x %x %x %x\n", *(unsigned int*)&v[0], *(unsigned int*)&v[1], *(unsigned int*)&v[2], *(unsigned int*)&v[3]);
}
#ifdef PARSEC_USE_SSE
void printvx(__m128 v)
{
    float* foo = (float*)&v;

    printf("%12x %12x %12x %12x\n", *(unsigned int*)&foo[0], *(unsigned int*)&foo[1], *(unsigned int*)&foo[2], *(unsigned int*)&foo[3]);
//    printf("%f %f %f %f\n", foo[0], foo[1], foo[2], foo[3]);
}

void printv(__m128 v)
{
    float* foo = (float*)&v;

    printf("%e %e %e %e\n", foo[0], foo[1], foo[2], foo[3]);
}
void printvix(__m128i v)
{
    unsigned int* foo = (unsigned int*)&v;

    printf("%x %x %x %x\n", *(unsigned int*)&foo[0], *(unsigned int*)&foo[1], *(unsigned int*)&foo[2], *(unsigned int*)&foo[3]);
//    printf("%f %f %f %f\n", foo[0], foo[1], foo[2], foo[3]);
}
void printvi(__m128i v)
{
    int* foo = (int*)&v;

    printf("%d %d %d %d\n", foo[0], foo[1], foo[2], foo[3]);
}
#endif

#ifdef PARSEC_USE_NEON_ASS
inline void printv(vsf4 v)
{
    float foo[4] = {0,0,0,0};
    __asm volatile (
        "vst1.32 {%q[v]}, [%[out]];"
        :
        : [v] "w" (v), [out] "r" (foo)
        : "memory"
    );

    printf("%12e %12e %12e %12e\n", foo[0], foo[1], foo[2], foo[3]);
}
#endif

#ifdef PARSEC_USE_NEON
#include <arm_neon.h>
void printv(float32x4_t v)
{
    float foo[4];

    vst1q_f32(foo, v);

    printf("%e %e %e %e\n", foo[0], foo[1], foo[2], foo[3]);
}

void printvi(int32x4_t v)
{
    int foo[4];

    vst1q_s32(foo, v);

    printf("%d %d %d %d\n", foo[0], foo[1], foo[2], foo[3]);
}

void printvu(uint32x4_t v)
{
    unsigned int foo[4];

    vst1q_u32(foo, v);

    printf("%u %u %u %u\n", foo[0], foo[1], foo[2], foo[3]);
}

void printvux(uint32x4_t v)
{
    unsigned int foo[4];

    vst1q_u32(foo, v);

    printf("%x %x %x %x\n", foo[0], foo[1], foo[2], foo[3]);
}
#endif

#ifdef PARSEC_USE_AVX
void printvv(__m256 v)
{
    float* foo = (float*)&v;

    printf("%f %f %f %f %f %f %f %f\n", foo[0], foo[1], foo[2], foo[3], foo[4], foo[5], foo[6], foo[7]);
}

void printvvi(__m256i v)
{
    int* foo = (int*)&v;

    printf("%d %d %d %d %d %d %d %d\n", foo[0], foo[1], foo[2], foo[3], foo[4], foo[5], foo[6], foo[7]);
}
#endif

#endif
