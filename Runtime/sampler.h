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

typedef enum {
    GSL_SAMPLER_PRESET_A = 0,
    GSL_SAMPLER_PRESET_B,
    GSL_SAMPLER_PRESET_C,
    GSL_SAMPLER_PRESET_NEAR_GREEDY,
    GSL_SAMPLER_PRESET_GREEDY,
    GSL_SAMPLER_PRESET_LEGACY,
    GSL_SAMPLER_PRESET_COUNT
} GSLSamplerPreset;

void gsl_sampler_initialize(GSLSampler *sampler, uint32_t seed);
void gsl_sampler_apply_preset(GSLSampler *sampler, GSLSamplerPreset preset);
const char *gsl_sampler_preset_name(GSLSamplerPreset preset);
uint16_t gsl_sampler_sample(GSLSampler *sampler, const float *logits,
                            size_t vocab_size, const uint16_t *history,
                            size_t history_count);

#ifdef __cplusplus
}
#endif

#endif
