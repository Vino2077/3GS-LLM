#include "decoder.h"

#include "int8_gemv.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define GSL_ROPE_PAIRS (GSL_HEAD_WIDTH / 2u)

struct GSLDecoder {
    const GSLModelWeights *weights;
    uint32_t position;
    float key_cache[GSL_LAYER_COUNT][GSL_CONTEXT_LENGTH][GSL_MODEL_WIDTH];
    float value_cache[GSL_LAYER_COUNT][GSL_CONTEXT_LENGTH][GSL_MODEL_WIDTH];
    float rope_cosine[GSL_CONTEXT_LENGTH][GSL_ROPE_PAIRS];
    float rope_sine[GSL_CONTEXT_LENGTH][GSL_ROPE_PAIRS];
    float hidden[GSL_MODEL_WIDTH];
    float normalized[GSL_MODEL_WIDTH];
    float query[GSL_MODEL_WIDTH];
    float key[GSL_MODEL_WIDTH];
    float value[GSL_MODEL_WIDTH];
    float attention[GSL_MODEL_WIDTH];
    float projected[GSL_MODEL_WIDTH];
    float gate[GSL_FFN_WIDTH];
    float up[GSL_FFN_WIDTH];
    float scores[GSL_CONTEXT_LENGTH];
    int8_t quantized[GSL_FFN_WIDTH];
    int32_t accumulators[GSL_VOCAB_SIZE];
};

static void gsl_rms_norm(const float *input, const float *weights,
                         float *output, size_t count)
{
    float sum = 0.0f;
    float scale;
    size_t index;
    for (index = 0u; index < count; ++index) {
        sum += input[index] * input[index];
    }
    scale = 1.0f / sqrtf(sum / (float)count + 1.0e-5f);
    for (index = 0u; index < count; ++index) {
        output[index] = input[index] * scale * weights[index];
    }
}

static float gsl_quantize_vector(const float *input, int8_t *output,
                                 size_t count)
{
    float maximum = 0.0f;
    float scale;
    float inverse;
    size_t index;
    for (index = 0u; index < count; ++index) {
        float magnitude = fabsf(input[index]);
        if (magnitude > maximum) {
            maximum = magnitude;
        }
    }
    if (maximum <= FLT_MIN) {
        memset(output, 0, count);
        return 1.0f;
    }
    scale = maximum / 127.0f;
    inverse = 1.0f / scale;
    for (index = 0u; index < count; ++index) {
        long value = lrintf(input[index] * inverse);
        if (value < -127L) {
            value = -127L;
        } else if (value > 127L) {
            value = 127L;
        }
        output[index] = (int8_t)value;
    }
    return scale;
}

static void gsl_q8_matvec(const GSLQ8Matrix *matrix,
                          const int8_t *quantized, float input_scale,
                          int32_t *accumulators, float *output)
{
    size_t row;
    gsl_gemv_i8(matrix->weights, quantized, accumulators,
                matrix->rows, matrix->columns);
    for (row = 0u; row < matrix->rows; ++row) {
        output[row] = (float)accumulators[row] *
                      matrix->row_scales[row] * input_scale;
    }
}

static void gsl_apply_rope(GSLDecoder *decoder, float *vector,
                           uint32_t position)
{
    size_t head;
    for (head = 0u; head < GSL_HEAD_COUNT; ++head) {
        float *head_vector = vector + head * GSL_HEAD_WIDTH;
        size_t pair;
        for (pair = 0u; pair < GSL_ROPE_PAIRS; ++pair) {
            float even = head_vector[pair * 2u];
            float odd = head_vector[pair * 2u + 1u];
            float cosine = decoder->rope_cosine[position][pair];
            float sine = decoder->rope_sine[position][pair];
            head_vector[pair * 2u] = even * cosine - odd * sine;
            head_vector[pair * 2u + 1u] = even * sine + odd * cosine;
        }
    }
}

static void gsl_attention(GSLDecoder *decoder, uint32_t layer,
                          uint32_t position)
{
    size_t head;
    for (head = 0u; head < GSL_HEAD_COUNT; ++head) {
        const float *query = decoder->query + head * GSL_HEAD_WIDTH;
        float *output = decoder->attention + head * GSL_HEAD_WIDTH;
        float maximum = -FLT_MAX;
        float denominator = 0.0f;
        size_t time;
        size_t dimension;

        for (time = 0u; time <= position; ++time) {
            const float *key = decoder->key_cache[layer][time] +
                               head * GSL_HEAD_WIDTH;
            float score = 0.0f;
            for (dimension = 0u; dimension < GSL_HEAD_WIDTH; ++dimension) {
                score += query[dimension] * key[dimension];
            }
            score *= 0.125f;
            decoder->scores[time] = score;
            if (score > maximum) {
                maximum = score;
            }
        }
        for (time = 0u; time <= position; ++time) {
            float probability = expf(decoder->scores[time] - maximum);
            decoder->scores[time] = probability;
            denominator += probability;
        }
        memset(output, 0, GSL_HEAD_WIDTH * sizeof(float));
        for (time = 0u; time <= position; ++time) {
            const float *value = decoder->value_cache[layer][time] +
                                 head * GSL_HEAD_WIDTH;
            float probability = decoder->scores[time] / denominator;
            for (dimension = 0u; dimension < GSL_HEAD_WIDTH; ++dimension) {
                output[dimension] += probability * value[dimension];
            }
        }
    }
}

GSLDecoder *gsl_decoder_create(const GSLModelWeights *weights)
{
    GSLDecoder *decoder;
    size_t position;
    size_t pair;
    if (weights == NULL) {
        return NULL;
    }
    decoder = (GSLDecoder *)malloc(sizeof(*decoder));
    if (decoder == NULL) {
        return NULL;
    }
    memset(decoder, 0, sizeof(*decoder));
    decoder->weights = weights;
    for (position = 0u; position < GSL_CONTEXT_LENGTH; ++position) {
        for (pair = 0u; pair < GSL_ROPE_PAIRS; ++pair) {
            float inverse = powf(10000.0f,
                                 -(float)(pair * 2u) / GSL_HEAD_WIDTH);
            float angle = (float)position * inverse;
            decoder->rope_cosine[position][pair] = cosf(angle);
            decoder->rope_sine[position][pair] = sinf(angle);
        }
    }
    return decoder;
}

void gsl_decoder_destroy(GSLDecoder *decoder)
{
    free(decoder);
}

void gsl_decoder_reset(GSLDecoder *decoder)
{
    if (decoder != NULL) {
        decoder->position = 0u;
    }
}

size_t gsl_decoder_memory_bytes(void)
{
    return sizeof(GSLDecoder);
}

int gsl_decoder_step(GSLDecoder *decoder, uint16_t token, float *logits)
{
    uint32_t layer;
    uint32_t position;
    float activation_scale;
    size_t index;

    if (decoder == NULL || logits == NULL || token >= GSL_VOCAB_SIZE ||
        decoder->position >= GSL_CONTEXT_LENGTH) {
        return 0;
    }
    position = decoder->position;
    {
        const int8_t *embedding = decoder->weights->embedding.weights +
                                  (size_t)token * GSL_MODEL_WIDTH;
        float scale = decoder->weights->embedding.row_scales[token];
        for (index = 0u; index < GSL_MODEL_WIDTH; ++index) {
            decoder->hidden[index] = (float)embedding[index] * scale;
        }
    }

    for (layer = 0u; layer < GSL_LAYER_COUNT; ++layer) {
        const GSLLayerWeights *weights = decoder->weights->layers + layer;
        gsl_rms_norm(decoder->hidden, weights->attention_norm,
                     decoder->normalized, GSL_MODEL_WIDTH);
        activation_scale = gsl_quantize_vector(
            decoder->normalized, decoder->quantized, GSL_MODEL_WIDTH
        );
        gsl_q8_matvec(&weights->query, decoder->quantized, activation_scale,
                      decoder->accumulators, decoder->query);
        gsl_q8_matvec(&weights->key, decoder->quantized, activation_scale,
                      decoder->accumulators, decoder->key);
        gsl_q8_matvec(&weights->value, decoder->quantized, activation_scale,
                      decoder->accumulators, decoder->value);
        gsl_apply_rope(decoder, decoder->query, position);
        gsl_apply_rope(decoder, decoder->key, position);
        memcpy(decoder->key_cache[layer][position], decoder->key,
               GSL_MODEL_WIDTH * sizeof(float));
        memcpy(decoder->value_cache[layer][position], decoder->value,
               GSL_MODEL_WIDTH * sizeof(float));
        gsl_attention(decoder, layer, position);
        activation_scale = gsl_quantize_vector(
            decoder->attention, decoder->quantized, GSL_MODEL_WIDTH
        );
        gsl_q8_matvec(&weights->attention_output, decoder->quantized,
                      activation_scale, decoder->accumulators,
                      decoder->projected);
        for (index = 0u; index < GSL_MODEL_WIDTH; ++index) {
            decoder->hidden[index] += decoder->projected[index];
        }

        gsl_rms_norm(decoder->hidden, weights->ffn_norm,
                     decoder->normalized, GSL_MODEL_WIDTH);
        activation_scale = gsl_quantize_vector(
            decoder->normalized, decoder->quantized, GSL_MODEL_WIDTH
        );
        gsl_q8_matvec(&weights->ffn_gate, decoder->quantized, activation_scale,
                      decoder->accumulators, decoder->gate);
        gsl_q8_matvec(&weights->ffn_up, decoder->quantized, activation_scale,
                      decoder->accumulators, decoder->up);
        for (index = 0u; index < GSL_FFN_WIDTH; ++index) {
            float gate = decoder->gate[index];
            decoder->gate[index] = gate / (1.0f + expf(-gate)) *
                                   decoder->up[index];
        }
        activation_scale = gsl_quantize_vector(
            decoder->gate, decoder->quantized, GSL_FFN_WIDTH
        );
        gsl_q8_matvec(&weights->ffn_down, decoder->quantized, activation_scale,
                      decoder->accumulators, decoder->projected);
        for (index = 0u; index < GSL_MODEL_WIDTH; ++index) {
            decoder->hidden[index] += decoder->projected[index];
        }
    }

    gsl_rms_norm(decoder->hidden, decoder->weights->final_norm,
                 decoder->normalized, GSL_MODEL_WIDTH);
    activation_scale = gsl_quantize_vector(
        decoder->normalized, decoder->quantized, GSL_MODEL_WIDTH
    );
    gsl_q8_matvec(&decoder->weights->embedding, decoder->quantized,
                  activation_scale, decoder->accumulators, logits);
    decoder->position = position + 1u;
    return 1;
}
