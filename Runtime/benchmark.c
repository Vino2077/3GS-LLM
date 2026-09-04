#include "benchmark.h"
#include "int8_gemv.h"

#include <mach/mach_time.h>
#include <stdlib.h>

static uint32_t gsl_random_state = 0x3A5F91C7u;

static uint32_t gsl_xorshift32(void)
{
    uint32_t x = gsl_random_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    gsl_random_state = x;
    return x;
}

static int8_t gsl_random_i8(void)
{
    return (int8_t)((int)(gsl_xorshift32() & 0xFFu) - 128);
}

static double gsl_elapsed_seconds(uint64_t start, uint64_t end)
{
    mach_timebase_info_data_t info;
    mach_timebase_info(&info);
    return ((double)(end - start) * (double)info.numer /
            (double)info.denom) / 1000000000.0;
}

int gsl_verify_int8_kernel(void)
{
    int8_t lhs[197];
    int8_t rhs[197];
    size_t i;

    for (i = 0; i < 197; ++i) {
        lhs[i] = gsl_random_i8();
        rhs[i] = gsl_random_i8();
    }

    return gsl_dot_i8(lhs, rhs, 197) ==
           gsl_dot_i8_reference(lhs, rhs, 197);
}

int gsl_benchmark_gemv(size_t rows, size_t cols, size_t iterations,
                       GSLBenchmarkResult *result)
{
    int8_t *matrix;
    int8_t *vector;
    int32_t *output;
    size_t matrix_count;
    size_t i;
    uint64_t start;
    uint64_t end;
    double elapsed;
    int32_t checksum = 0;

    if (result == NULL || rows == 0 || cols == 0 || iterations == 0) {
        return 0;
    }

    matrix_count = rows * cols;
    matrix = (int8_t *)malloc(matrix_count);
    vector = (int8_t *)malloc(cols);
    output = (int32_t *)malloc(rows * sizeof(int32_t));
    if (matrix == NULL || vector == NULL || output == NULL) {
        free(output);
        free(vector);
        free(matrix);
        return 0;
    }

    for (i = 0; i < matrix_count; ++i) {
        matrix[i] = gsl_random_i8();
    }
    for (i = 0; i < cols; ++i) {
        vector[i] = gsl_random_i8();
    }

    gsl_gemv_i8(matrix, vector, output, rows, cols);
    start = mach_absolute_time();
    for (i = 0; i < iterations; ++i) {
        gsl_gemv_i8(matrix, vector, output, rows, cols);
        checksum ^= output[i % rows];
    }
    end = mach_absolute_time();

    elapsed = gsl_elapsed_seconds(start, end);
    result->rows = rows;
    result->cols = cols;
    result->iterations = iterations;
    result->milliseconds_per_iteration =
        elapsed * 1000.0 / (double)iterations;
    result->mmac_per_second =
        ((double)rows * (double)cols * (double)iterations) /
        elapsed / 1000000.0;
    result->checksum = checksum;

    free(output);
    free(vector);
    free(matrix);
    return 1;
}
