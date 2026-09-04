#ifndef GSL_TOKENIZER_H
#define GSL_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint16_t left;
    uint16_t right;
    uint16_t result;
} GSLTokenizerMerge;

typedef struct {
    uint32_t vocab_size;
    uint32_t merge_count;
    uint16_t special_ids[6];
    const uint16_t *base_ids;
    const uint32_t *decoder_offsets;
    const uint8_t *decoder_bytes;
    const GSLTokenizerMerge *merges;
    uint32_t *lookup_keys;
    uint32_t *lookup_ranks;
    uint16_t *lookup_results;
    size_t lookup_capacity;
} GSLTokenizer;

int gsl_tokenizer_open(GSLTokenizer *tokenizer, const void *bytes, size_t length);
void gsl_tokenizer_close(GSLTokenizer *tokenizer);

/* Encodes one piece produced by the standard ByteLevel regular expression. */
size_t gsl_tokenizer_encode_piece(const GSLTokenizer *tokenizer,
                                  const uint8_t *utf8, size_t length,
                                  uint16_t *output, size_t capacity);

size_t gsl_tokenizer_decode(const GSLTokenizer *tokenizer,
                            const uint16_t *tokens, size_t count,
                            uint8_t *output, size_t capacity);

#ifdef __cplusplus
}
#endif

#endif
