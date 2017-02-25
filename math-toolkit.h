#ifndef __RAY_MATH_TOOLKIT_H
#define __RAY_MATH_TOOLKIT_H

#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <immintrin.h>

static inline __attribute__ ((always_inline))
void normalize(double *v)
{
    register __m256d p = _mm256_set_pd(v[0], v[1], v[2], 0.0);
    register __m256d mul = _mm256_mul_pd(p, p);
    register __m256d hadd = _mm256_hadd_pd(mul, mul);
    register __m128d tmp1 = _mm_set1_pd(hadd[3]);
    register __m128d tmp2 = _mm_set1_pd(hadd[1]);
    register __m128d cross_product_result = _mm_add_pd(tmp1, tmp2);
    register __m128d d = _mm_sqrt_pd(cross_product_result);
    assert(d[1] != 0.0 && "Error calculating normal");

    register __m256d d1 = _mm256_set_pd(v[0], v[1], v[2], 0.0);
    register __m256d d2 = _mm256_set_pd(d[1], d[1], d[1], 0.0);
    register __m256d div_result = _mm256_div_pd(d1, d2);
    v[0] = div_result[3];
    v[1] = div_result[2];
    v[2] = div_result[1];
}

static inline __attribute__ ((always_inline))
double length(const double *v)
{
    register __m256d p = _mm256_set_pd(v[0], v[1], v[2], 0.0);
    register __m256d mul = _mm256_mul_pd(p, p);
    register __m256d hadd = _mm256_hadd_pd(mul, mul);
    register __m128d r1 = _mm_set1_pd(hadd[3]);
    register __m128d r2 = _mm_set1_pd(hadd[1]);
    register __m128d result = _mm_add_pd(r2, r1);
    return _mm_sqrt_pd(result)[1];
}

static inline __attribute__ ((always_inline))
void add_vector(const double *a, const double *b, double *out)
{
    register __m256d p1 = _mm256_set_pd(a[0], a[1], a[2], 0.0);
    register __m256d p2 = _mm256_set_pd(b[0], b[1], b[2], 0.0);
    register __m256d result = _mm256_add_pd(p1, p2);
    out[0] = result[3];
    out[1] = result[2];
    out[2] = result[1];
}

static inline __attribute__ ((always_inline))
void subtract_vector(const double *a, const double *b, double *out)
{
    register __m256d p1 = _mm256_set_pd(a[0], a[1], a[2], 0.0);
    register __m256d p2 = _mm256_set_pd(b[0], b[1], b[2], 0.0);
    register __m256d result = _mm256_sub_pd(p1, p2);
    out[0] = result[3];
    out[1] = result[2];
    out[2] = result[1];
}

static inline __attribute__ ((always_inline))
void multiply_vectors(const double *a, const double *b, double *out)
{
    register __m256d p1 = _mm256_set_pd(a[0], a[1], a[2], 0.0);
    register __m256d p2 = _mm256_set_pd(b[0], b[1], b[2], 0.0);
    register __m256d result = _mm256_mul_pd(p1, p2);
    out[0] = result[3];
    out[1] = result[2];
    out[2] = result[1];
}

static inline __attribute__ ((always_inline))
void multiply_vector(const double *a, double b, double *out)
{
    register __m256d p1 = _mm256_set_pd(a[0], a[1], a[2], 0.0);
    register __m256d p2 = _mm256_set_pd(b, b, b, 0.0);
    register __m256d result = _mm256_mul_pd(p1, p2);
    out[0] = result[3];
    out[1] = result[2];
    out[2] = result[1];
}

static inline __attribute__ ((always_inline))
void cross_product(const double *v1, const double *v2, double *out)
{
    register __m256d p1 = _mm256_set_pd(v1[1], v1[2], v1[0], 0.0);
    register __m256d p2 = _mm256_set_pd(v1[2], v1[0], v1[1], 0.0);
    register __m256d p3 = _mm256_set_pd(v2[2], v2[0], v2[1], 0.0);
    register __m256d p4 = _mm256_set_pd(v2[1], v2[2], v2[0], 0.0);
    register __m256d r1 = _mm256_mul_pd(p1, p3);
    register __m256d r2 = _mm256_mul_pd(p2, p4);
    register __m256d result = _mm256_sub_pd(r1, r2);

    out[0] = result[3];
    out[1] = result[2];
    out[2] = result[1];
}

static inline __attribute__ ((always_inline))
double dot_product(const double *v1, const double *v2)
{
    register __m256d p1 = _mm256_set_pd(v1[0], v1[1], v1[2], 0.0);
    register __m256d p2 = _mm256_set_pd(v2[0], v2[1], v2[2], 0.0);
    register __m256d mul = _mm256_mul_pd(p1, p2);
    register __m256d hadd = _mm256_hadd_pd(mul, mul);
    register __m128d result1 = _mm_set1_pd(hadd[3]);
    register __m128d result2 = _mm_set1_pd(hadd[1]);
    return  _mm_add_pd(result1, result2)[1];
}

static inline
void scalar_triple_product(const double *u, const double *v, const double *w,
                           double *out)
{
    cross_product(v, w, out);
    multiply_vectors(u, out, out);
}

static inline
double scalar_triple(const double *u, const double *v, const double *w)
{
    double tmp[3];
    cross_product(w, u, tmp);
    return dot_product(v, tmp);
}

#endif
