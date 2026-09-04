#ifndef GSL_MODEL_FORMAT_H
#define GSL_MODEL_FORMAT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GSL_VOCAB_SIZE 8192u
#define GSL_CONTEXT_LENGTH 256u
#define GSL_MODEL_WIDTH 384u
#define GSL_LAYER_COUNT 8u
#define GSL_HEAD_COUNT 6u
#define GSL_HEAD_WIDTH 64u
#define GSL_FFN_WIDTH 1024u

typedef struct {
    uint32_t rows;
    uint32_t columns;
    const int8_t *weights;
    const float *row_scales;
} GSLQ8Matrix;

typedef struct {
    const float *attention_norm;
    GSLQ8Matrix query;
    GSLQ8Matrix key;
    GSLQ8Matrix value;
    GSLQ8Matrix attention_output;
    const float *ffn_norm;
    GSLQ8Matrix ffn_gate;
    GSLQ8Matrix ffn_up;
    GSLQ8Matrix ffn_down;
} GSLLayerWeights;

typedef struct {
    GSLQ8Matrix embedding;
    GSLLayerWeights layers[GSL_LAYER_COUNT];
    const float *final_norm;
    const uint8_t *container;
    size_t container_length;
} GSLModelWeights;

int gsl_model_weights_open(GSLModelWeights *model,
                           const void *bytes, size_t length);

#ifdef __cplusplus
}
#endif

#endif
