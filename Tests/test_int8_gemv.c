#include "../Runtime/int8_gemv.h"

#include <stdio.h>

int main(void)
{
    int8_t lhs[197];
    int8_t rhs[197];
    int8_t matrix[3 * 197];
    int32_t output[3];
    size_t i;

    for (i = 0; i < 197; ++i) {
        lhs[i] = (int8_t)((int)((i * 17u + 5u) % 255u) - 127);
        rhs[i] = (int8_t)((int)((i * 29u + 9u) % 255u) - 127);
    }

    if (gsl_dot_i8(lhs, rhs, 197) != gsl_dot_i8_reference(lhs, rhs, 197)) {
        fputs("dot product mismatch\n", stderr);
        return 1;
    }

    for (i = 0; i < 3 * 197; ++i) {
        matrix[i] = (int8_t)((int)((i * 11u + 3u) % 255u) - 127);
    }
    gsl_gemv_i8(matrix, rhs, output, 3, 197);
    for (i = 0; i < 3; ++i) {
        int32_t expected = gsl_dot_i8_reference(matrix + i * 197, rhs, 197);
        if (output[i] != expected) {
            fprintf(stderr, "row %lu mismatch\n", (unsigned long)i);
            return 1;
        }
    }

    puts("int8 GEMV tests passed");
    return 0;
}
