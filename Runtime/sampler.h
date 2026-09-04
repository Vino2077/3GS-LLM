#ifndef GSL_SAMPLER_H
#define GSL_SAMPLER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t random_state;
    float temperature;
    float repetition_penalty;
    size_t top_k;
} GSLSampler;

void gsl_sampler_initialize(GSLSampler *sampler, uint32_t seed);
uint16_t gsl_sampler_sample(GSLSampler *sampler, const float *logits,
                            size_t vocab_size, const uint16_t *history,
                            size_t history_count);

#ifdef __cplusplus
}
#endif

#endif
