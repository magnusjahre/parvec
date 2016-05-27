// NEON log-exp made by Hallgeir Lien (hallgeir.lien@gmail.com)

#ifndef ARM_MATHFUN_H_IN
#define ARM_MATHFUN_H_IN

//#ifdef NEON_INTRIN

#include <arm_neon.h>

static inline __attribute__((always_inline)) float32x4_t log128_neon_intrin(float32x4_t x)
{
    //compute invalid mask
    uint32x4_t invalid_mask = vcltq_f32(x, vdupq_n_f32(0.f)),
               tmp2, mask;
    int32x4_t  tmp1;

    float32x4_t e, y, z, tmp1f;

    //cut off denormalized stuff
    x = vmaxq_f32(x, vreinterpretq_f32_u32(vdupq_n_u32(0x00800000)));

    //Shift left by 23 bits
    tmp1 = vshrq_n_s32(vreinterpretq_s32_f32(x), 23);

    //Bitwise-AND with inverse mantissa mask
    //(Keep only fractional part)
    tmp2 = vandq_u32(vreinterpretq_u32_f32(x), vdupq_n_u32(0x807fffff));
    tmp2 = vorrq_u32(tmp2, vreinterpretq_u32_f32(vdupq_n_f32(0.5f)));
    x = vreinterpretq_f32_u32(tmp2);

    tmp1 = vsubq_s32(tmp1, vdupq_n_s32(0x7f));
    e = vcvtq_f32_s32(tmp1);

    e = vaddq_f32(e, vdupq_n_f32(1.0f));

    mask = vcltq_f32(x, vdupq_n_f32(0.707106781186547524f));
    tmp2 = vandq_u32(vreinterpretq_u32_f32(x), mask);
    x = vsubq_f32(x, vdupq_n_f32(1.0f));
    e = vsubq_f32(e, vreinterpretq_f32_u32(vandq_u32(vreinterpretq_u32_f32(vdupq_n_f32(1.0f)), mask)));
    x = vaddq_f32(x, vreinterpretq_f32_u32(tmp2));

    z = vmulq_f32(x,x);

    y = vdupq_n_f32(7.0376836292E-2);   //p0
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(-1.1514610310E-1)); //p1
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(1.1676998740E-1));  //p2
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(-1.2420140846E-1)); //p3
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(1.4249322787E-1));  //p4
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(-1.6668057665E-1)); //p5
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(2.0000714765E-1));  //p6
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(-2.4999993993E-1)); //p7
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(3.3333331174E-1)); //p8
    y = vmulq_f32(y, x);

    y = vmulq_f32(y, z);

    tmp1f = vmulq_f32(e, vdupq_n_f32(-2.12194440e-4));
    y = vaddq_f32(y, tmp1f);

    tmp1f = vmulq_f32(z, vdupq_n_f32(0.5f));
    y = vsubq_f32(y, tmp1f);

    tmp1f = vmulq_f32(e, vdupq_n_f32(0.693359375));
    x = vaddq_f32(x, y);
    x = vaddq_f32(x, tmp1f);
    x = vreinterpretq_f32_u32(vorrq_u32(vreinterpretq_u32_f32(x), invalid_mask));

    return x;
}

static inline __attribute__((always_inline)) float32x4_t exp128_neon_intrin(float32x4_t x)
{
    float32x4_t fx, tmp1f, y, z;
    uint32x4_t mask;
    int32x4_t tmp1i;

    x = vminq_f32(x, vdupq_n_f32(88.3762626647949f));
    x = vmaxq_f32(x, vdupq_n_f32(-88.3762626647949f));
    fx = vmulq_f32(x, vdupq_n_f32(1.44269504088896341));
    fx = vaddq_f32(fx, vdupq_n_f32(0.5));

    tmp1f = vcvtq_f32_s32(vcvtq_s32_f32(fx));
    mask = vcgtq_f32(tmp1f, fx);
    mask = vandq_u32(mask, vreinterpretq_u32_f32(vdupq_n_f32(1.0f)));

    fx = vsubq_f32(tmp1f, vreinterpretq_f32_u32(mask));

    tmp1f = vmulq_f32(fx, vdupq_n_f32(0.693359375));
    z = vmulq_f32(fx, vdupq_n_f32(-2.12194440e-4f));
    x = vsubq_f32(x, tmp1f);
    x = vsubq_f32(x, z);

    z = vmulq_f32(x,x);

    y = vdupq_n_f32(1.9875691500E-4f); //p0
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(1.3981999507E-3f));//p1
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(8.3334519073E-3f));//p2
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(4.1665795894E-2f));//p3
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(1.6666665459E-1f));//p4
    y = vmulq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(5.0000001201E-1f));//p5
    y = vmulq_f32(y, z);
    y = vaddq_f32(y, x);
    y = vaddq_f32(y, vdupq_n_f32(1.0f));

    tmp1i = vcvtq_s32_f32(fx);
    tmp1i = vaddq_s32(tmp1i, vdupq_n_s32(0x7f));
    tmp1i = vshlq_n_s32(tmp1i, 23);

    tmp1f = vreinterpretq_f32_s32(tmp1i);
    x = vmulq_f32(y, tmp1f);

    return x;
}
//#endif

#endif
