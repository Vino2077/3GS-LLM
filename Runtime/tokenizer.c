#include "tokenizer.h"

#include <limits.h>
#include <stdlib.h>
#include <string.h>

#define GSL_TOKENIZER_HEADER_SIZE 128u
#define GSL_TOKENIZER_VERSION 1u

#if defined(__GNUC__)
#define GSL_PACKED __attribute__((packed))
#else
#define GSL_PACKED
#endif

typedef struct GSL_PACKED {
    char magic[8];
    uint32_t version;
    uint32_t vocab_size;
    uint32_t merge_count;
    uint32_t special_ids[6];
    uint32_t decoded_bytes;
    uint8_t payload_sha256[32];
} GSLTokenizerHeader;

static size_t gsl_align(size_t value, size_t alignment)
{
    return (value + alignment - 1u) & ~(alignment - 1u);
}

static size_t gsl_next_power_of_two(size_t value)
{
    size_t result = 1u;
    while (result < value) {
        if (result > SIZE_MAX / 2u) {
            return 0u;
        }
        result *= 2u;
    }
    return result;
}

static size_t gsl_hash_pair(uint32_t key, size_t mask)
{
    return ((size_t)(key * UINT32_C(2654435761))) & mask;
}

static int gsl_insert_merge(GSLTokenizer *tokenizer, uint32_t key,
                            uint32_t rank, uint16_t result)
{
    size_t mask = tokenizer->lookup_capacity - 1u;
    size_t index = gsl_hash_pair(key, mask);
    while (tokenizer->lookup_ranks[index] != UINT32_MAX) {
        if (tokenizer->lookup_keys[index] == key) {
            return 0;
        }
        index = (index + 1u) & mask;
    }
    tokenizer->lookup_keys[index] = key;
    tokenizer->lookup_ranks[index] = rank;
    tokenizer->lookup_results[index] = result;
    return 1;
}

static int gsl_find_merge(const GSLTokenizer *tokenizer, uint16_t left,
                          uint16_t right, uint32_t *rank, uint16_t *result)
{
    uint32_t key = ((uint32_t)left << 16u) | (uint32_t)right;
    size_t mask = tokenizer->lookup_capacity - 1u;
    size_t index = gsl_hash_pair(key, mask);
    while (tokenizer->lookup_ranks[index] != UINT32_MAX) {
        if (tokenizer->lookup_keys[index] == key) {
            *rank = tokenizer->lookup_ranks[index];
            *result = tokenizer->lookup_results[index];
            return 1;
        }
        index = (index + 1u) & mask;
    }
    return 0;
}

void gsl_tokenizer_close(GSLTokenizer *tokenizer)
{
    if (tokenizer == NULL) {
        return;
    }
    free(tokenizer->lookup_keys);
    free(tokenizer->lookup_ranks);
    free(tokenizer->lookup_results);
    memset(tokenizer, 0, sizeof(*tokenizer));
}

int gsl_tokenizer_open(GSLTokenizer *tokenizer, const void *bytes, size_t length)
{
    const GSLTokenizerHeader *header;
    const uint8_t *body;
    size_t offsets_bytes;
    size_t merge_offset;
    size_t expected_length;
    size_t rank;

    if (tokenizer == NULL || bytes == NULL || length < GSL_TOKENIZER_HEADER_SIZE) {
        return 0;
    }
    memset(tokenizer, 0, sizeof(*tokenizer));
    header = (const GSLTokenizerHeader *)bytes;
    if (memcmp(header->magic, "3GSTOK1", 7u) != 0 ||
        header->version != GSL_TOKENIZER_VERSION ||
        header->vocab_size != 8192u || header->merge_count != 7930u) {
        return 0;
    }
    for (rank = 0u; rank < 6u; ++rank) {
        if (header->special_ids[rank] != rank) {
            return 0;
        }
        tokenizer->special_ids[rank] = (uint16_t)rank;
    }

    offsets_bytes = ((size_t)header->vocab_size + 1u) * sizeof(uint32_t);
    merge_offset = 256u * sizeof(uint16_t) + offsets_bytes +
                   (size_t)header->decoded_bytes;
    merge_offset = gsl_align(merge_offset, 2u);
    expected_length = GSL_TOKENIZER_HEADER_SIZE + merge_offset +
                      (size_t)header->merge_count * sizeof(GSLTokenizerMerge);
    if (expected_length != length) {
        return 0;
    }

    body = (const uint8_t *)bytes + GSL_TOKENIZER_HEADER_SIZE;
    tokenizer->vocab_size = header->vocab_size;
    tokenizer->merge_count = header->merge_count;
    tokenizer->base_ids = (const uint16_t *)body;
    tokenizer->decoder_offsets =
        (const uint32_t *)(body + 256u * sizeof(uint16_t));
    tokenizer->decoder_bytes = body + 256u * sizeof(uint16_t) + offsets_bytes;
    tokenizer->merges = (const GSLTokenizerMerge *)(body + merge_offset);
    if (tokenizer->decoder_offsets[0] != 0u ||
        tokenizer->decoder_offsets[tokenizer->vocab_size] !=
            header->decoded_bytes) {
        gsl_tokenizer_close(tokenizer);
        return 0;
    }

    tokenizer->lookup_capacity =
        gsl_next_power_of_two((size_t)header->merge_count * 2u);
    if (tokenizer->lookup_capacity == 0u) {
        gsl_tokenizer_close(tokenizer);
        return 0;
    }
    tokenizer->lookup_keys =
        (uint32_t *)malloc(tokenizer->lookup_capacity * sizeof(uint32_t));
    tokenizer->lookup_ranks =
        (uint32_t *)malloc(tokenizer->lookup_capacity * sizeof(uint32_t));
    tokenizer->lookup_results =
        (uint16_t *)malloc(tokenizer->lookup_capacity * sizeof(uint16_t));
    if (tokenizer->lookup_keys == NULL || tokenizer->lookup_ranks == NULL ||
        tokenizer->lookup_results == NULL) {
        gsl_tokenizer_close(tokenizer);
        return 0;
    }
    for (rank = 0u; rank < tokenizer->lookup_capacity; ++rank) {
        tokenizer->lookup_ranks[rank] = UINT32_MAX;
    }
    for (rank = 0u; rank < tokenizer->merge_count; ++rank) {
        const GSLTokenizerMerge *merge = tokenizer->merges + rank;
        uint32_t key = ((uint32_t)merge->left << 16u) | merge->right;
        if (!gsl_insert_merge(tokenizer, key, (uint32_t)rank, merge->result)) {
            gsl_tokenizer_close(tokenizer);
            return 0;
        }
    }
    return 1;
}

size_t gsl_tokenizer_encode_piece(const GSLTokenizer *tokenizer,
                                  const uint8_t *utf8, size_t length,
                                  uint16_t *output, size_t capacity)
{
    uint16_t *working;
    size_t count;
    size_t index;

    if (tokenizer == NULL || utf8 == NULL || output == NULL ||
        capacity < length) {
        return 0u;
    }
    if (length == 0u) {
        return 0u;
    }
    working = output;
    for (index = 0u; index < length; ++index) {
        working[index] = tokenizer->base_ids[utf8[index]];
    }
    count = length;

    while (count > 1u) {
        uint32_t best_rank = UINT32_MAX;
        uint16_t best_result = 0u;
        size_t best_index = 0u;
        int found = 0;
        for (index = 0u; index + 1u < count; ++index) {
            uint32_t rank;
            uint16_t result;
            if (gsl_find_merge(tokenizer, working[index], working[index + 1u],
                               &rank, &result) && rank < best_rank) {
                best_rank = rank;
                best_result = result;
                best_index = index;
                found = 1;
            }
        }
        if (!found) {
            break;
        }
        working[best_index] = best_result;
        memmove(working + best_index + 1u, working + best_index + 2u,
                (count - best_index - 2u) * sizeof(uint16_t));
        --count;
    }
    return count;
}

size_t gsl_tokenizer_decode(const GSLTokenizer *tokenizer,
                            const uint16_t *tokens, size_t count,
                            uint8_t *output, size_t capacity)
{
    size_t required = 0u;
    size_t index;
    if (tokenizer == NULL || tokens == NULL) {
        return 0u;
    }
    for (index = 0u; index < count; ++index) {
        uint16_t token = tokens[index];
        uint32_t start;
        uint32_t end;
        size_t token_length;
        if (token >= tokenizer->vocab_size) {
            return 0u;
        }
        start = tokenizer->decoder_offsets[token];
        end = tokenizer->decoder_offsets[token + 1u];
        if (end < start) {
            return 0u;
        }
        token_length = (size_t)(end - start);
        if (output != NULL && required + token_length <= capacity) {
            memcpy(output + required, tokenizer->decoder_bytes + start,
                   token_length);
        }
        required += token_length;
    }
    return required;
}
