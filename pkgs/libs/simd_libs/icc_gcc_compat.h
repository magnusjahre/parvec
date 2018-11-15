#ifndef __SIMD_COMPAT_H
#define __SIMD_COMPAT_H

//For compatibility between different compilers



//GCC does not have the intrinsics for specific comparisons
#ifdef __GNUC__
#if defined (PARSEC_USE_AVX) || defined (PARSEC_USE_AVX512)
#define _mm256_cmple_ps(a,b) _mm256_cmp_ps(a,b,_CMP_LE_OS)
#define _mm256_cmpge_ps(a,b) _mm256_cmp_ps(b,a,_CMP_LE_OS)
#define _mm256_cmplt_ps(a,b) _mm256_cmp_ps(a,b,_CMP_LT_OS)
#define _mm256_cmpgt_ps(a,b) _mm256_cmp_ps(b,a,_CMP_LT_OS)

#define _mm256_cmpeq_ps(a,b) _mm256_cmp_ps(a,b,_CMP_EQ_OQ)
#define _mm256_cmpneq_ps(a,b) _mm256_cmp_ps(a,b,_CMP_NEQ_OQ)

#define _mm256_cmple_pd(a,b) _mm256_cmp_pd(a,b,_CMP_LE_OS)
#define _mm256_cmpge_pd(a,b) _mm256_cmp_pd(b,a,_CMP_LE_OS)
#define _mm256_cmplt_pd(a,b) _mm256_cmp_pd(a,b,_CMP_LT_OS)
#define _mm256_cmpgt_pd(a,b) _mm256_cmp_pd(b,a,_CMP_LT_OS)

#define _mm256_cmpneq_pd(a,b) _mm256_cmp_pd(a,b,_CMP_NEQ_OQ)
#define _mm256_cmpeq_pd(a,b) _mm256_cmp_pd(a,b,_CMP_EQ_OQ)

#endif // AVX or AVX512

#if defined (PARSEC_USE_AVX512)

#ifndef _CMP_EQ_OQ
/* Equal (ordered, non-signaling)  */
#define _CMP_EQ_OQ	0x00
/* Less-than (ordered, signaling)  */
#define _CMP_LT_OS	0x01
/* Less-than-or-equal (ordered, signaling)  */
#define _CMP_LE_OS	0x02
/* Unordered (non-signaling)  */
#define _CMP_UNORD_Q	0x03
/* Not-equal (unordered, non-signaling)  */
#define _CMP_NEQ_UQ	0x04
/* Not-less-than (unordered, signaling)  */
#define _CMP_NLT_US	0x05
/* Not-less-than-or-equal (unordered, signaling)  */
#define _CMP_NLE_US	0x06
/* Ordered (nonsignaling)   */
#define _CMP_ORD_Q	0x07
/* Equal (unordered, non-signaling)  */
#define _CMP_EQ_UQ	0x08
/* Not-greater-than-or-equal (unordered, signaling)  */
#define _CMP_NGE_US	0x09
/* Not-greater-than (unordered, signaling)  */
#define _CMP_NGT_US	0x0a
/* False (ordered, non-signaling)  */
#define _CMP_FALSE_OQ	0x0b
/* Not-equal (ordered, non-signaling)  */
#define _CMP_NEQ_OQ	0x0c
/* Greater-than-or-equal (ordered, signaling)  */
#define _CMP_GE_OS	0x0d
/* Greater-than (ordered, signaling)  */
#define _CMP_GT_OS	0x0e
/* True (unordered, non-signaling)  */
#define _CMP_TRUE_UQ	0x0f
/* Equal (ordered, signaling)  */
#define _CMP_EQ_OS	0x10
/* Less-than (ordered, non-signaling)  */
#define _CMP_LT_OQ	0x11
/* Less-than-or-equal (ordered, non-signaling)  */
#define _CMP_LE_OQ	0x12
/* Unordered (signaling)  */
#define _CMP_UNORD_S	0x13
/* Not-equal (unordered, signaling)  */
#define _CMP_NEQ_US	0x14
/* Not-less-than (unordered, non-signaling)  */
#define _CMP_NLT_UQ	0x15
/* Not-less-than-or-equal (unordered, non-signaling)  */
#define _CMP_NLE_UQ	0x16
/* Ordered (signaling)  */
#define _CMP_ORD_S	0x17
/* Equal (unordered, signaling)  */
#define _CMP_EQ_US	0x18
/* Not-greater-than-or-equal (unordered, non-signaling)  */
#define _CMP_NGE_UQ	0x19
/* Not-greater-than (unordered, non-signaling)  */
#define _CMP_NGT_UQ	0x1a
/* False (ordered, signaling)  */
#define _CMP_FALSE_OS	0x1b
/* Not-equal (ordered, signaling)  */
#define _CMP_NEQ_OS	0x1c
/* Greater-than-or-equal (ordered, non-signaling)  */
#define _CMP_GE_OQ	0x1d
/* Greater-than (ordered, non-signaling)  */
#define _CMP_GT_OQ	0x1e
/* True (unordered, signaling)  */
#define _CMP_TRUE_US	0x1f
#endif

#define _mm512_cmple_ps(a,b) _mm512_castsi512_ps(_mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(a,b,_CMP_LE_OS),0xFFFFFFFF))
#define _mm512_cmpge_ps(a,b) _mm512_castsi512_ps(_mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(b,a,_CMP_LE_OS),0xFFFFFFFF))
#define _mm512_cmplt_ps(a,b) _mm512_castsi512_ps(_mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(a,b,_CMP_LT_OS),0xFFFFFFFF))
#define _mm512_cmpgt_ps(a,b) _mm512_castsi512_ps(_mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(b,a,_CMP_LT_OS),0xFFFFFFFF))
#define _mm512_cmpeq_ps(a,b) _mm512_castsi512_ps(_mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(a,b,_CMP_EQ_OQ),0xFFFFFFFF))
#define _mm512_cmpneq_ps(a,b) _mm512_castsi512_ps(_mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(a,b,_CMP_NEQ_OQ),0xFFFFFFFF))

#define _mm512_cmpeq_epi32(a,b) _mm512_maskz_set1_epi32(_mm512_cmp_epi32_mask(a,b,_CMP_EQ_OQ),0xFFFFFFFF)

#define _mm512_cmple_pd(a,b) _mm512_castsi512_pd(_mm512_maskz_set1_epi64(_mm512_cmp_pd_mask(a,b,_CMP_LE_OS),0xFFFFFFFFFFFFFFFF))
#define _mm512_cmpge_pd(a,b) _mm512_castsi512_pd(_mm512_maskz_set1_epi64(_mm512_cmp_pd_mask(b,a,_CMP_LE_OS),0xFFFFFFFFFFFFFFFF))
#define _mm512_cmplt_pd(a,b) _mm512_castsi512_pd(_mm512_maskz_set1_epi64(_mm512_cmp_pd_mask(a,b,_CMP_LT_OS),0xFFFFFFFFFFFFFFFF))
#define _mm512_cmpgt_pd(a,b) _mm512_castsi512_pd(_mm512_maskz_set1_epi64(_mm512_cmp_pd_mask(b,a,_CMP_LT_OS),0xFFFFFFFFFFFFFFFF))
#define _mm512_cmpneq_pd(a,b) _mm512_castsi512_pd(_mm512_maskz_set1_epi64(_mm512_cmp_pd_mask(a,b,_CMP_NEQ_OQ),0xFFFFFFFFFFFFFFFF))
#define _mm512_cmpeq_pd(a,b) _mm512_castsi512_pd(_mm512_maskz_set1_epi64(_mm512_cmp_pd_mask(a,b,_CMP_EQ_OQ),0xFFFFFFFFFFFFFFFF))

#define _mm512_cmpeq_epi64(a,b) _mm512_maskz_set1_epi64(_mm512_cmp_epi64_mask(a,b,_CMP_EQ_OQ),0xFFFFFFFFFFFFFFFF)

#define _custom_mm512_cmple_ps_mask(a,b) _mm512_cmp_ps_mask(a,b,_CMP_LE_OS)
#define _custom_mm512_cmpge_ps_mask(a,b) _mm512_cmp_ps_mask(b,a,_CMP_LE_OS)
#define _custom_mm512_cmplt_ps_mask(a,b) _mm512_cmp_ps_mask(a,b,_CMP_LT_OS)
#define _custom_mm512_cmpgt_ps_mask(a,b) _mm512_cmp_ps_mask(b,a,_CMP_LT_OS)
#define _custom_mm512_cmpeq_ps_mask(a,b) _mm512_cmp_ps_mask(a,b,_CMP_EQ_OQ)
#define _custom_mm512_cmpneq_ps_mask(a,b)_mm512_cmp_ps_mask(a,b,_CMP_NEQ_OQ)

#define _custom_mm512_cmple_pd_mask(a,b) _mm512_cmp_pd_mask(a,b,_CMP_LE_OS)
#define _custom_mm512_cmpge_pd_mask(a,b) _mm512_cmp_pd_mask(b,a,_CMP_LE_OS)
#define _custom_mm512_cmplt_pd_mask(a,b) _mm512_cmp_pd_mask(a,b,_CMP_LT_OS)
#define _custom_mm512_cmpgt_pd_mask(a,b) _mm512_cmp_pd_mask(b,a,_CMP_LT_OS)
#define _custom_mm512_cmpneq_pd_mask(a,b) _mm512_cmp_pd_mask(a,b,_CMP_NEQ_OQ)
#define _custom_mm512_cmpeq_pd_mask(a,b) _mm512_cmp_pd_mask(a,b,_CMP_EQ_OQ)

#endif // AVX512
#endif // __GNUC__

#endif
