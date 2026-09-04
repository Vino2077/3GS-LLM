#include "int8_gemv.h"

#if defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

int gsl_int8_neon_enabled(void)
{
#if defined(__ARM_NEON__)
    return 1;
#else
    return 0;
#endif
}

int32_t gsl_dot_i8_reference(const int8_t *lhs, const int8_t *rhs, size_t count)
{
    int32_t sum = 0;
    size_t i;
    for (i = 0; i < count; ++i) {
        sum += (int32_t)lhs[i] * (int32_t)rhs[i];
    }
    return sum;
}

int32_t gsl_dot_i8(const int8_t *lhs, const int8_t *rhs, size_t count)
{
#if defined(__ARM_NEON__)
    int32x4_t sum = vdupq_n_s32(0);
    size_t i = 0;

    for (; i + 15 < count; i += 16) {
        int8x16_t a = vld1q_s8(lhs + i);
        int8x16_t b = vld1q_s8(rhs + i);
        int16x8_t low = vmull_s8(vget_low_s8(a), vget_low_s8(b));
        int16x8_t high = vmull_s8(vget_high_s8(a), vget_high_s8(b));
        sum = vaddq_s32(sum, vpaddlq_s16(low));
        sum = vaddq_s32(sum, vpaddlq_s16(high));
    }

    {
        int32_t lanes[4];
        int32_t total;
        vst1q_s32(lanes, sum);
        total = lanes[0] + lanes[1] + lanes[2] + lanes[3];
        for (; i < count; ++i) {
            total += (int32_t)lhs[i] * (int32_t)rhs[i];
        }
        return total;
    }
#else
    return gsl_dot_i8_reference(lhs, rhs, count);
#endif
}

void gsl_gemv_i8(const int8_t *matrix, const int8_t *vector,
                  int32_t *output, size_t rows, size_t cols)
{
    size_t row;
    for (row = 0; row < rows; ++row) {
        output[row] = gsl_dot_i8(matrix + row * cols, vector, cols);
    }
}
