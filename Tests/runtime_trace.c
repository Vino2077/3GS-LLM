#include "../Runtime/decoder.h"

#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TRACE_TOP_K 10u

static unsigned char *read_file(const char *path, size_t *length)
{
    FILE *source = fopen(path, "rb");
    unsigned char *bytes;
    long size;
    if (source == NULL) {
        return NULL;
    }
    if (fseek(source, 0, SEEK_END) != 0) {
        fclose(source);
        return NULL;
    }
    size = ftell(source);
    if (size <= 0 || fseek(source, 0, SEEK_SET) != 0) {
        fclose(source);
        return NULL;
    }
    bytes = (unsigned char *)malloc((size_t)size);
    if (bytes == NULL || fread(bytes, 1u, (size_t)size, source) != (size_t)size) {
        free(bytes);
        fclose(source);
        return NULL;
    }
    fclose(source);
    *length = (size_t)size;
    return bytes;
}

static size_t parse_tokens(char *text, uint16_t *tokens, size_t capacity)
{
    size_t count = 0u;
    char *cursor = text;
    if (*cursor == '\0') {
        return 0u;
    }
    while (*cursor != '\0' && count < capacity) {
        char *end;
        unsigned long value = strtoul(cursor, &end, 10);
        if (end == cursor || value >= GSL_VOCAB_SIZE) {
            return SIZE_MAX;
        }
        tokens[count++] = (uint16_t)value;
        if (*end == '\0') {
            break;
        }
        if (*end != ',') {
            return SIZE_MAX;
        }
        cursor = end + 1;
    }
    return count;
}

static void print_top(size_t step, const float *logits)
{
    float values[TRACE_TOP_K];
    uint16_t tokens[TRACE_TOP_K];
    size_t index;
    size_t token;
    for (index = 0u; index < TRACE_TOP_K; ++index) {
        values[index] = -FLT_MAX;
        tokens[index] = 0u;
    }
    for (token = 0u; token < GSL_VOCAB_SIZE; ++token) {
        size_t insertion;
        float value = logits[token];
        if (value <= values[TRACE_TOP_K - 1u]) {
            continue;
        }
        insertion = TRACE_TOP_K - 1u;
        while (insertion > 0u && value > values[insertion - 1u]) {
            values[insertion] = values[insertion - 1u];
            tokens[insertion] = tokens[insertion - 1u];
            --insertion;
        }
        values[insertion] = value;
        tokens[insertion] = (uint16_t)token;
    }
    printf("{\"step\":%lu,\"top\":[", (unsigned long)step);
    for (index = 0u; index < TRACE_TOP_K; ++index) {
        printf("%s{\"id\":%u,\"logit\":%.9g}", index ? "," : "",
               (unsigned)tokens[index], values[index]);
    }
    printf("]}\n");
}

int main(int argc, char **argv)
{
    unsigned char *container;
    size_t container_length = 0u;
    GSLModelWeights weights;
    GSLDecoder *decoder;
    uint16_t prefix[GSL_CONTEXT_LENGTH];
    uint16_t forced[GSL_CONTEXT_LENGTH];
    size_t prefix_count;
    size_t forced_count;
    float *logits;
    size_t index;
    int result = 1;
    if (argc != 4) {
        fprintf(stderr, "usage: runtime_trace model.bin prefix_ids forced_ids\n");
        return 2;
    }
    container = read_file(argv[1], &container_length);
    prefix_count = parse_tokens(argv[2], prefix, GSL_CONTEXT_LENGTH);
    forced_count = parse_tokens(argv[3], forced, GSL_CONTEXT_LENGTH);
    if (container == NULL || prefix_count == SIZE_MAX || forced_count == SIZE_MAX ||
        prefix_count == 0u || prefix_count + forced_count > GSL_CONTEXT_LENGTH) {
        fprintf(stderr, "invalid input\n");
        free(container);
        return 2;
    }
    if (!gsl_model_weights_open(&weights, container, container_length)) {
        fprintf(stderr, "invalid model container\n");
        free(container);
        return 2;
    }
    decoder = gsl_decoder_create(&weights);
    logits = (float *)malloc(GSL_VOCAB_SIZE * sizeof(float));
    if (decoder == NULL || logits == NULL) {
        fprintf(stderr, "allocation failed\n");
        goto cleanup;
    }
    for (index = 0u; index < prefix_count; ++index) {
        if (!gsl_decoder_step(decoder, prefix[index], logits)) {
            fprintf(stderr, "prefix decode failed\n");
            goto cleanup;
        }
    }
    for (index = 0u; index <= forced_count; ++index) {
        print_top(index, logits);
        if (index < forced_count &&
            !gsl_decoder_step(decoder, forced[index], logits)) {
            fprintf(stderr, "forced decode failed\n");
            goto cleanup;
        }
    }
    result = 0;

cleanup:
    free(logits);
    gsl_decoder_destroy(decoder);
    free(container);
    return result;
}
