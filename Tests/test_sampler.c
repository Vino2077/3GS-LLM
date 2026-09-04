#include "../Runtime/sampler.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

static int close_enough(float first, float second)
{
    return fabsf(first - second) < 1.0e-6f;
}

int main(void)
{
    GSLSampler sampler;
    float logits[] = {0.0f, 1.0f, 5.0f, 2.0f};
    gsl_sampler_initialize(&sampler, 123u);
    assert(close_enough(sampler.temperature, 0.6f));
    assert(sampler.top_k == 20u);
    assert(close_enough(sampler.repetition_penalty, 1.02f));

    gsl_sampler_apply_preset(&sampler, GSL_SAMPLER_PRESET_A);
    assert(close_enough(sampler.temperature, 0.45f));
    assert(sampler.top_k == 10u);
    assert(close_enough(sampler.repetition_penalty, 1.0f));

    gsl_sampler_apply_preset(&sampler, GSL_SAMPLER_PRESET_B);
    assert(close_enough(sampler.temperature, 0.25f));
    assert(sampler.top_k == 5u);

    gsl_sampler_apply_preset(&sampler, GSL_SAMPLER_PRESET_C);
    assert(close_enough(sampler.temperature, 0.6f));
    assert(sampler.top_k == 20u);
    assert(close_enough(sampler.repetition_penalty, 1.02f));

    gsl_sampler_apply_preset(&sampler, GSL_SAMPLER_PRESET_GREEDY);
    assert(gsl_sampler_sample(&sampler, logits, 4u, NULL, 0u) == 2u);
    assert(gsl_sampler_preset_name(GSL_SAMPLER_PRESET_NEAR_GREEDY)[0] == 'N');
    puts("sampler preset test passed");
    return 0;
}
