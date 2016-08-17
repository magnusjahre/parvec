#ifndef __SIMD_COMPAT_H
#define __SIMD_COMPAT_H

//For compatibility between different compilers



//GCC does not have the intrinsics for specific comparisons
#ifdef __GNUC__
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


#define _mm512_cmple_ps(a,b) _mm512_castsi512_ps(_mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(a,b,_CMP_LE_OS),0xFFFFFFFF))
#define _mm512_cmpge_ps(a,b) _mm512_castsi512_ps(_mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(b,a,_CMP_LE_OS),0xFFFFFFFF))
#define _mm512_cmplt_ps(a,b) _mm512_castsi512_ps(_mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(a,b,_CMP_LT_OS),0xFFFFFFFF))
#define _mm512_cmpgt_ps(a,b) _mm512_castsi512_ps(_mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(b,a,_CMP_LT_OS),0xFFFFFFFF))
#define _mm512_cmpeq_ps(a,b) _mm512_castsi512_ps(_mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(a,b,_CMP_EQ_OQ),0xFFFFFFFF))
#define _mm512_cmpneq_ps(a,b) _mm512_castsi512_ps(_mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(a,b,_CMP_NEQ_OQ),0xFFFFFFFF))

#define _mm512_cmpeq_epi32(a,b) _mm512_maskz_set1_epi32(_mm512_cmp_epi32_mask(a,b,_CMP_EQ_OQ),0xFFFFFFFF)

#define _mm512_cmple_pd(a,b) _mm512_castsi512_pd(_mm512_maskz_set1_epi32(_mm512_cmp_pd_mask(a,b,_CMP_LE_OS),0xFFFFFFFFFFFFFFFF))
#define _mm512_cmpge_pd(a,b) _mm512_castsi512_pd(_mm512_maskz_set1_epi32(_mm512_cmp_pd_mask(b,a,_CMP_LE_OS),0xFFFFFFFFFFFFFFFF))
#define _mm512_cmplt_pd(a,b) _mm512_castsi512_pd(_mm512_maskz_set1_epi32(_mm512_cmp_pd_mask(a,b,_CMP_LT_OS),0xFFFFFFFFFFFFFFFF))
#define _mm512_cmpgt_pd(a,b) _mm512_castsi512_pd(_mm512_maskz_set1_epi32(_mm512_cmp_pd_mask(b,a,_CMP_LT_OS),0xFFFFFFFFFFFFFFFF))
#define _mm512_cmpneq_pd(a,b) _mm512_castsi512_pd(_mm512_maskz_set1_epi32(_mm512_cmp_pd_mask(a,b,_CMP_NEQ_OQ),0xFFFFFFFFFFFFFFFF))
#define _mm512_cmpeq_pd(a,b) _mm512_castsi512_pd(_mm512_maskz_set1_epi32(_mm512_cmp_pd_mask(a,b,_CMP_EQ_OQ),0xFFFFFFFFFFFFFFFF))

#define _mm512_cmpeq_epi64(a,b) _mm512_maskz_set1_epi64(_mm512_cmp_epi64_mask(a,b,_CMP_EQ_OQ),0xFFFFFFFFFFFFFFFF)

#define _mm512_cmple_ps_mask(a,b) _mm512_cmp_ps_mask(a,b,_CMP_LE_OS)
#define _mm512_cmpge_ps_mask(a,b) _mm512_cmp_ps_mask(b,a,_CMP_LE_OS)
#define _mm512_cmplt_ps_mask(a,b) _mm512_cmp_ps_mask(a,b,_CMP_LT_OS)
#define _mm512_cmpgt_ps_mask(a,b) _mm512_cmp_ps_mask(b,a,_CMP_LT_OS)
#define _mm512_cmpeq_ps_mask(a,b) _mm512_cmp_ps_mask(a,b,_CMP_EQ_OQ)
#define _mm512_cmpneq_ps_mask(a,b)_mm512_cmp_ps_mask(a,b,_CMP_NEQ_OQ)

#define _mm512_cmple_pd_mask(a,b) _mm512_cmp_pd_mask(a,b,_CMP_LE_OS)
#define _mm512_cmpge_pd_mask(a,b) _mm512_cmp_pd_mask(b,a,_CMP_LE_OS)
#define _mm512_cmplt_pd_mask(a,b) _mm512_cmp_pd_mask(a,b,_CMP_LT_OS)
#define _mm512_cmpgt_pd_mask(a,b) _mm512_cmp_pd_mask(b,a,_CMP_LT_OS)
#define _mm512_cmpneq_pd_mask(a,b) _mm512_cmp_pd_mask(a,b,_CMP_NEQ_OQ)
#define _mm512_cmpeq_pd_mask(a,b) _mm512_cmp_pd_mask(a,b,_CMP_EQ_OQ)

#endif // __GNUC__

#endif
