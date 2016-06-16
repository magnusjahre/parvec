#ifndef __AVX_COMPAT_H
#define __AVX_COMPAT_H

//For compatibility between different compilers



//GCC does not have the intrinsics for specific comparisons
#ifdef __GNUC__
#define _mm256_cmple_ps(a,b) _mm256_cmp_ps(a,b,2)
#define _mm256_cmpge_ps(a,b) _mm256_cmp_ps(b,a,2)
#define _mm256_cmplt_ps(a,b) _mm256_cmp_ps(a,b,1)
#define _mm256_cmpgt_ps(a,b) _mm256_cmp_ps(b,a,1)

#define _mm256_cmpeq_ps(a,b) _mm256_cmp_ps(a,b,_CMP_EQ_OQ)
#define _mm256_cmpneq_ps(a,b) _mm256_cmp_ps(a,b,_CMP_NEQ_OQ)

#define _mm256_cmple_pd(a,b) _mm256_cmp_pd(a,b,2)
#define _mm256_cmpge_pd(a,b) _mm256_cmp_pd(b,a,2)
#define _mm256_cmplt_pd(a,b) _mm256_cmp_pd(a,b,1)
#define _mm256_cmpgt_pd(a,b) _mm256_cmp_pd(b,a,1)

#define _mm256_cmpneq_pd(a,b) _mm256_cmp_pd(a,b,_CMP_NEQ_OQ)
#define _mm256_cmpeq_pd(a,b) _mm256_cmp_pd(a,b,_CMP_EQ_OQ)

#endif

#endif
