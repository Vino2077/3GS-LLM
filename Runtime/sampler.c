#include "sampler.h"

#include <float.h>
#include <math.h>

#define GSL_MAX_TOP_K 64u

static uint32_t gsl_random(GSLSampler *sampler)
{
    uint32_t value = sampler->random_state;
    value ^= value << 13u;
    value ^= value >> 17u;
    value ^= value << 5u;
    sampler->random_state = value;
    return value;
}

static int gsl_was_seen(uint16_t token, const uint16_t *history,
                        size_t history_count)
{
    size_t index;
    size_t start = history_count > 64u ? history_count - 64u : 0u;
    for (index = start; index < history_count; ++index) {
        if (history[index] == token) {
            return 1;
        }
    }
    return 0;
}

void gsl_sampler_initialize(GSLSampler *sampler, uint32_t seed)
{
    sampler->random_state = seed == 0u ? UINT32_C(0x6d2b79f5) : seed;
    gsl_sampler_apply_preset(sampler, GSL_SAMPLER_PRESET_C);
}

void gsl_sampler_apply_preset(GSLSampler *sampler, GSLSamplerPreset preset)
{
    if (sampler == NULL) {
        return;
    }
    switch (preset) {
    case GSL_SAMPLER_PRESET_A:
        sampler->temperature = 0.45f;
        sampler->top_k = 10u;
        sampler->repetition_penalty = 1.0f;
        break;
    case GSL_SAMPLER_PRESET_B:
        sampler->temperature = 0.25f;
        sampler->top_k = 5u;
        sampler->repetition_penalty = 1.0f;
        break;
    case GSL_SAMPLER_PRESET_C:
        sampler->temperature = 0.6f;
        sampler->top_k = 20u;
        sampler->repetition_penalty = 1.02f;
        break;
    case GSL_SAMPLER_PRESET_NEAR_GREEDY:
        sampler->temperature = 0.1f;
        sampler->top_k = 3u;
        sampler->repetition_penalty = 1.0f;
        break;
    case GSL_SAMPLER_PRESET_GREEDY:
        sampler->temperature = 1.0f;
        sampler->top_k = 1u;
        sampler->repetition_penalty = 1.0f;
        break;
    case GSL_SAMPLER_PRESET_LEGACY:
    default:
        sampler->temperature = 0.8f;
        sampler->top_k = 40u;
        sampler->repetition_penalty = 1.08f;
        break;
    }
}

const char *gsl_sampler_preset_name(GSLSamplerPreset preset)
{
    switch (preset) {
    case GSL_SAMPLER_PRESET_A:
        return "A";
    case GSL_SAMPLER_PRESET_B:
        return "B";
    case GSL_SAMPLER_PRESET_C:
        return "C";
    case GSL_SAMPLER_PRESET_NEAR_GREEDY:
        return "NG";
    case GSL_SAMPLER_PRESET_GREEDY:
        return "G";
    case GSL_SAMPLER_PRESET_LEGACY:
        return "L";
    default:
        return "?";
    }
}

uint16_t gsl_sampler_sample(GSLSampler *sampler, const float *logits,
                            size_t vocab_size, const uint16_t *history,
                            size_t history_count)
{
    float best_scores[GSL_MAX_TOP_K];
    uint16_t best_tokens[GSL_MAX_TOP_K];
    size_t top_k;
    size_t token;
    size_t index;
    float maximum;
    float total = 0.0f;
    float target;

    if (sampler == NULL || logits == NULL || vocab_size == 0u ||
        vocab_size > 65535u) {
        return 0u;
    }
    top_k = sampler->top_k;
    if (top_k == 0u || top_k > GSL_MAX_TOP_K) {
        top_k = GSL_MAX_TOP_K;
    }
    if (top_k > vocab_size) {
        top_k = vocab_size;
    }
    for (index = 0u; index < top_k; ++index) {
        best_scores[index] = -FLT_MAX;
        best_tokens[index] = 0u;
    }

    for (token = 0u; token < vocab_size; ++token) {
        float score = logits[token];
        size_t insertion;
        if (history != NULL && gsl_was_seen((uint16_t)token, history,
                                             history_count)) {
            score = score >= 0.0f
                        ? score / sampler->repetition_penalty
                        : score * sampler->repetition_penalty;
        }
        if (score <= best_scores[top_k - 1u]) {
            continue;
        }
        insertion = top_k - 1u;
        while (insertion > 0u && score > best_scores[insertion - 1u]) {
            best_scores[insertion] = best_scores[insertion - 1u];
            best_tokens[insertion] = best_tokens[insertion - 1u];
            --insertion;
        }
        best_scores[insertion] = score;
        best_tokens[insertion] = (uint16_t)token;
    }

    maximum = best_scores[0];
    for (index = 0u; index < top_k; ++index) {
        best_scores[index] = expf(
            (best_scores[index] - maximum) / sampler->temperature
        );
        total += best_scores[index];
    }
    target = ((float)(gsl_random(sampler) >> 8u) / 16777216.0f) * total;
    for (index = 0u; index + 1u < top_k; ++index) {
        if (target <= best_scores[index]) {
            return best_tokens[index];
        }
        target -= best_scores[index];
    }
    return best_tokens[top_k - 1u];
}
