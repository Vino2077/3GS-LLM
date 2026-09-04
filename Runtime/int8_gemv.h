#ifndef GSL_INT8_GEMV_H
#define GSL_INT8_GEMV_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int gsl_int8_neon_enabled(void);
int32_t gsl_dot_i8_reference(const int8_t *lhs, const int8_t *rhs, size_t count);
int32_t gsl_dot_i8(const int8_t *lhs, const int8_t *rhs, size_t count);
void gsl_gemv_i8(const int8_t *matrix, const int8_t *vector,
                  int32_t *output, size_t rows, size_t cols);

#ifdef __cplusplus
}
#endif

#endif
