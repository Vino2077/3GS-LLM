#include "model_format.h"

#include <stdio.h>
#include <string.h>

#define GSL_MODEL_HEADER_SIZE 128u
#define GSL_MODEL_RECORD_SIZE 36u
#define GSL_MODEL_RECORD_ALIGNMENT 64u
#define GSL_MODEL_TENSOR_COUNT 74u
#define GSL_TYPE_Q8_ROWWISE 1u
#define GSL_TYPE_FLOAT32 2u

typedef struct {
    const char *name;
    uint8_t type;
    uint8_t dimensions;
    uint32_t shape[2];
    const uint8_t *data;
    const uint8_t *auxiliary;
} GSLParsedTensor;

static uint16_t gsl_load_u16(const uint8_t *bytes)
{
    return (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8u);
}

static uint32_t gsl_load_u32(const uint8_t *bytes)
{
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8u) |
           ((uint32_t)bytes[2] << 16u) | ((uint32_t)bytes[3] << 24u);
}

static uint64_t gsl_load_u64(const uint8_t *bytes)
{
    return (uint64_t)gsl_load_u32(bytes) |
           ((uint64_t)gsl_load_u32(bytes + 4u) << 32u);
}

static size_t gsl_align(size_t value, size_t alignment)
{
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static int gsl_name_equals(const uint8_t *name, uint16_t length,
                           const char *expected)
{
    size_t expected_length = strlen(expected);
    return expected_length == length && memcmp(name, expected, length) == 0;
}

static int gsl_parse_tensor(const uint8_t *container, size_t length,
                            size_t *offset, const char *expected_name,
                            uint8_t expected_type, uint32_t first,
                            uint32_t second, GSLParsedTensor *tensor)
{
    const uint8_t *record;
    const uint8_t *name;
    uint16_t name_length;
    uint8_t dimensions;
    uint32_t shape0;
    uint32_t shape1;
    uint64_t data_bytes;
    uint64_t auxiliary_bytes;
    uint64_t expected_data;
    uint64_t expected_auxiliary;
    size_t data_offset;
    size_t end_offset;

    if (*offset > length || length - *offset < GSL_MODEL_RECORD_SIZE) {
        return 0;
    }
    record = container + *offset;
    name_length = gsl_load_u16(record);
    dimensions = record[3];
    shape0 = gsl_load_u32(record + 4u);
    shape1 = gsl_load_u32(record + 8u);
    data_bytes = gsl_load_u64(record + 20u);
    auxiliary_bytes = gsl_load_u64(record + 28u);
    if (record[2] != expected_type || dimensions != (second == 0u ? 1u : 2u) ||
        shape0 != first || shape1 != second) {
        return 0;
    }

    name = record + GSL_MODEL_RECORD_SIZE;
    if ((size_t)name_length > length - (*offset + GSL_MODEL_RECORD_SIZE) ||
        !gsl_name_equals(name, name_length, expected_name)) {
        return 0;
    }
    data_offset = gsl_align(
        *offset + GSL_MODEL_RECORD_SIZE + (size_t)name_length,
        GSL_MODEL_RECORD_ALIGNMENT
    );
    expected_data = (uint64_t)first * (uint64_t)(second == 0u ? 4u : second);
    expected_auxiliary = expected_type == GSL_TYPE_Q8_ROWWISE
                             ? (uint64_t)first * sizeof(float)
                             : 0u;
    if (data_bytes != expected_data || auxiliary_bytes != expected_auxiliary ||
        data_bytes > SIZE_MAX || auxiliary_bytes > SIZE_MAX ||
        data_offset > length || (size_t)data_bytes > length - data_offset) {
        return 0;
    }
    end_offset = data_offset + (size_t)data_bytes;
    if ((size_t)auxiliary_bytes > length - end_offset) {
        return 0;
    }

    tensor->name = expected_name;
    tensor->type = expected_type;
    tensor->dimensions = dimensions;
    tensor->shape[0] = shape0;
    tensor->shape[1] = shape1;
    tensor->data = container + data_offset;
    tensor->auxiliary = tensor->data + (size_t)data_bytes;
    *offset = gsl_align(
        end_offset + (size_t)auxiliary_bytes,
        GSL_MODEL_RECORD_ALIGNMENT
    );
    return *offset <= length;
}

static void gsl_set_matrix(GSLQ8Matrix *matrix,
                           const GSLParsedTensor *tensor)
{
    matrix->rows = tensor->shape[0];
    matrix->columns = tensor->shape[1];
    matrix->weights = (const int8_t *)tensor->data;
    matrix->row_scales = (const float *)tensor->auxiliary;
}

int gsl_model_weights_open(GSLModelWeights *model,
                           const void *bytes, size_t length)
{
    const uint8_t *container = (const uint8_t *)bytes;
    size_t offset = GSL_MODEL_HEADER_SIZE;
    uint32_t layer;
    GSLParsedTensor tensor;
    char name[80];

    if (model == NULL || container == NULL || length < GSL_MODEL_HEADER_SIZE) {
        return 0;
    }
    memset(model, 0, sizeof(*model));
    if (memcmp(container, "3GSLLM1", 7u) != 0 ||
        gsl_load_u32(container + 8u) != 1u ||
        gsl_load_u32(container + 12u) != GSL_VOCAB_SIZE ||
        gsl_load_u32(container + 16u) != GSL_CONTEXT_LENGTH ||
        gsl_load_u32(container + 20u) != GSL_MODEL_WIDTH ||
        gsl_load_u32(container + 24u) != GSL_LAYER_COUNT ||
        gsl_load_u32(container + 28u) != GSL_HEAD_COUNT ||
        gsl_load_u32(container + 32u) != GSL_FFN_WIDTH ||
        gsl_load_u32(container + 36u) != GSL_MODEL_TENSOR_COUNT) {
        return 0;
    }

    if (!gsl_parse_tensor(container, length, &offset,
                          "token_embedding.weight", GSL_TYPE_Q8_ROWWISE,
                          GSL_VOCAB_SIZE, GSL_MODEL_WIDTH, &tensor)) {
        return 0;
    }
    gsl_set_matrix(&model->embedding, &tensor);

    for (layer = 0u; layer < GSL_LAYER_COUNT; ++layer) {
        GSLLayerWeights *weights = model->layers + layer;
#define GSL_PARSE_NAME(suffix, type, rows, columns)                         \
        snprintf(name, sizeof(name), "layers.%lu.%s",                     \
                 (unsigned long)layer, suffix);                            \
        if (!gsl_parse_tensor(container, length, &offset, name, type,      \
                              rows, columns, &tensor)) {                    \
            return 0;                                                       \
        }

        GSL_PARSE_NAME("attention_norm.weight", GSL_TYPE_FLOAT32,
                       GSL_MODEL_WIDTH, 0u)
        weights->attention_norm = (const float *)tensor.data;
        GSL_PARSE_NAME("attention.q_proj.weight", GSL_TYPE_Q8_ROWWISE,
                       GSL_MODEL_WIDTH, GSL_MODEL_WIDTH)
        gsl_set_matrix(&weights->query, &tensor);
        GSL_PARSE_NAME("attention.k_proj.weight", GSL_TYPE_Q8_ROWWISE,
                       GSL_MODEL_WIDTH, GSL_MODEL_WIDTH)
        gsl_set_matrix(&weights->key, &tensor);
        GSL_PARSE_NAME("attention.v_proj.weight", GSL_TYPE_Q8_ROWWISE,
                       GSL_MODEL_WIDTH, GSL_MODEL_WIDTH)
        gsl_set_matrix(&weights->value, &tensor);
        GSL_PARSE_NAME("attention.o_proj.weight", GSL_TYPE_Q8_ROWWISE,
                       GSL_MODEL_WIDTH, GSL_MODEL_WIDTH)
        gsl_set_matrix(&weights->attention_output, &tensor);
        GSL_PARSE_NAME("ffn_norm.weight", GSL_TYPE_FLOAT32,
                       GSL_MODEL_WIDTH, 0u)
        weights->ffn_norm = (const float *)tensor.data;
        GSL_PARSE_NAME("feed_forward.gate_proj.weight", GSL_TYPE_Q8_ROWWISE,
                       GSL_FFN_WIDTH, GSL_MODEL_WIDTH)
        gsl_set_matrix(&weights->ffn_gate, &tensor);
        GSL_PARSE_NAME("feed_forward.up_proj.weight", GSL_TYPE_Q8_ROWWISE,
                       GSL_FFN_WIDTH, GSL_MODEL_WIDTH)
        gsl_set_matrix(&weights->ffn_up, &tensor);
        GSL_PARSE_NAME("feed_forward.down_proj.weight", GSL_TYPE_Q8_ROWWISE,
                       GSL_MODEL_WIDTH, GSL_FFN_WIDTH)
        gsl_set_matrix(&weights->ffn_down, &tensor);
#undef GSL_PARSE_NAME
    }
    if (!gsl_parse_tensor(container, length, &offset, "final_norm.weight",
                          GSL_TYPE_FLOAT32, GSL_MODEL_WIDTH, 0u, &tensor) ||
        offset != length) {
        return 0;
    }
    model->final_norm = (const float *)tensor.data;
    model->container = container;
    model->container_length = length;
    return 1;
}
