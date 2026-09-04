#ifndef GSL_BENCHMARK_H
#define GSL_BENCHMARK_H

#include <stddef.h>
#include <stdint.h>

#include "int8_gemv.h"

typedef struct {
    size_t rows;
    size_t cols;
    size_t iterations;
    double milliseconds_per_iteration;
    double mmac_per_second;
    int32_t checksum;
} GSLBenchmarkResult;

int gsl_verify_int8_kernel(void);
int gsl_benchmark_gemv(size_t rows, size_t cols, size_t iterations,
                       GSLBenchmarkResult *result);

#endif
