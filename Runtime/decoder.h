#ifndef GSL_DECODER_H
#define GSL_DECODER_H

#include "model_format.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct GSLDecoder GSLDecoder;

GSLDecoder *gsl_decoder_create(const GSLModelWeights *weights);
void gsl_decoder_destroy(GSLDecoder *decoder);
void gsl_decoder_reset(GSLDecoder *decoder);
size_t gsl_decoder_memory_bytes(void);

/* Consumes one token and writes logits for the following token. */
int gsl_decoder_step(GSLDecoder *decoder, uint16_t token, float *logits);

#ifdef __cplusplus
}
#endif

#endif
