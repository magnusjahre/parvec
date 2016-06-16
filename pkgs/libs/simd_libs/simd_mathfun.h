// Macro based EXP and LOG path functions
// Juan M. Cebrian, NTNU / BSC  - 2016.
//

#ifndef __SIMD_MATHFUN__
#define __SIMD_MATHFUN__

static inline _MM_TYPE simd_log(_MM_TYPE x) {
  _MM_TYPE_I emm0;
  _MM_TYPE one = _MM_SET(1.0);
  _MM_TYPE half = _MM_SET(0.5);

  _MM_TYPE invalid_mask = _MM_CMPLE(x, _MM_SETZERO());
  x = _MM_MAX(x, _MM_CAST_I_TO_FP(_MM_SET_I(_MM_MINNORMPOS)));  /* cut off denormalized stuff */
  emm0 = _MM_SRLI_I(_MM_CAST_FP_TO_I(x), _MM_MANTISSA_BITS);

  /* keep only the fractional part */
  x = _MM_AND(x, _MM_CAST_I_TO_FP(_MM_SET_I(_MM_MANTISSA_MASK)));
  x = _MM_OR(x, half);

  emm0 = _MM_SUB_I(emm0, _MM_SET_I(_MM_EXP_BIAS));

  _MM_TYPE e = _MM_CVT_I_TO_FP(emm0);

  e = _MM_ADD(e, one);

  _MM_TYPE mask = _MM_CMPLT(x, _MM_SET(0.707106781186547524));
  _MM_TYPE tmp = _MM_AND(x, mask);
  x = _MM_SUB(x, one);
  e = _MM_SUB(e, _MM_AND(one, mask));
  x = _MM_ADD(x, tmp);

  _MM_TYPE z = _MM_MUL(x,x);
  _MM_TYPE y = _MM_SET(7.0376836292E-2);
  y = _MM_MUL(y, x);
  y = _MM_ADD(y, _MM_SET(-1.1514610310E-1));
  y = _MM_MUL(y, x);
  y = _MM_ADD(y, _MM_SET(1.1676998740E-1));
  y = _MM_MUL(y, x);
  y = _MM_ADD(y, _MM_SET(1.2420140846E-1));
  y = _MM_MUL(y, x);
  y = _MM_ADD(y, _MM_SET(+1.4249322787E-1));
  y = _MM_MUL(y, x);
  y = _MM_ADD(y, _MM_SET(-1.6668057665E-1));
  y = _MM_MUL(y, x);
  y = _MM_ADD(y, _MM_SET(+2.0000714765E-1));
  y = _MM_MUL(y, x);
  y = _MM_ADD(y, _MM_SET(-2.4999993993E-1));
  y = _MM_MUL(y, x);
  y = _MM_ADD(y, _MM_SET(+3.3333331174E-1));
  y = _MM_MUL(y, x);

  y = _MM_MUL(y, z);

  tmp = _MM_MUL(e, _MM_SET(-2.12194440e-4));
  y = _MM_ADD(y, tmp);


  tmp = _MM_MUL(z, half);
  y = _MM_SUB(y, tmp);

  tmp = _MM_MUL(e, _MM_SET(0.693359375));
  x = _MM_ADD(x, y);
  x = _MM_ADD(x, tmp);
  x = _MM_OR(x, invalid_mask); // negative arg will be NAN
  return x;
}

static inline _MM_TYPE simd_exp(_MM_TYPE x) {
  _MM_TYPE tmp = _MM_SETZERO(), fx;
  _MM_TYPE_I emm0 = _MM_SETZERO_I();
  _MM_TYPE one = _MM_SET(1.0);
  _MM_TYPE half = _MM_SET(0.5);

  x = _MM_MIN(x, _MM_SET(88.3762626647949));
  x = _MM_MAX(x, _MM_SET(-88.3762626647949));

  /* express exp(x) as exp(g + n*log(2)) */
  fx = _MM_MUL(x, _MM_SET(1.44269504088896341));
  fx = _MM_ADD(fx, half);

  tmp = _MM_FLOOR(fx);

  /* if greater, substract 1 */
  _MM_TYPE mask = _MM_CMPGT(tmp, fx);
  mask = _MM_AND(mask, one);
  fx = _MM_SUB(tmp, mask);

  tmp = _MM_MUL(fx, _MM_SET(0.693359375));
  _MM_TYPE z = _MM_MUL(fx, _MM_SET(-2.12194440e-4));
  x = _MM_SUB(x, tmp);
  x = _MM_SUB(x, z);

  z = _MM_MUL(x,x);

  _MM_TYPE y = _MM_SET(1.9875691500E-4);
  y = _MM_MUL(y, x);
  y = _MM_ADD(y, _MM_SET(1.3981999507E-3));
  y = _MM_MUL(y, x);
  y = _MM_ADD(y, _MM_SET(8.3334519073E-3));
  y = _MM_MUL(y, x);
  y = _MM_ADD(y, _MM_SET(4.1665795894E-2));
  y = _MM_MUL(y, x);
  y = _MM_ADD(y, _MM_SET(1.6666665459E-1));
  y = _MM_MUL(y, x);
  y = _MM_ADD(y, _MM_SET(5.0000001201E-1));
  y = _MM_MUL(y, z);
  y = _MM_ADD(y, x);
  y = _MM_ADD(y, one);

  /* build 2^n */
  //
  emm0 = _MM_CVT_FP_TO_I(fx);
  emm0 = _MM_ADD_I(emm0, _MM_SET_I(_MM_EXP_BIAS));
  emm0 = _MM_SLLI_I(emm0, _MM_MANTISSA_BITS);

  _MM_TYPE pow2n = _MM_CAST_I_TO_FP(emm0);
  y = _MM_MUL(y, pow2n);

  return y;
}

#endif // __SIMD_MATHFUN__
