#include "../Runtime/tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    FILE *file;
    long file_length;
    unsigned char *container;
    GSLTokenizer tokenizer;
    const unsigned char probe[] = {
        0xd0, 0x9f, 0xd1, 0x80, 0xd0, 0xb8,
        0xd0, 0xb2, 0xd0, 0xb5, 0xd1, 0x82
    }; /* "Привет" in UTF-8. */
    uint16_t tokens[sizeof(probe)];
    unsigned char decoded[sizeof(probe)];
    size_t token_count;
    size_t decoded_count;

    if (argc != 2) {
        fprintf(stderr, "usage: %s tokenizer.bin\n", argv[0]);
        return 2;
    }
    file = fopen(argv[1], "rb");
    if (file == NULL || fseek(file, 0, SEEK_END) != 0) {
        return 1;
    }
    file_length = ftell(file);
    if (file_length <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        return 1;
    }
    container = (unsigned char *)malloc((size_t)file_length);
    if (container == NULL ||
        fread(container, 1u, (size_t)file_length, file) != (size_t)file_length) {
        return 1;
    }
    fclose(file);

    if (!gsl_tokenizer_open(&tokenizer, container, (size_t)file_length)) {
        fputs("could not open tokenizer\n", stderr);
        return 1;
    }
    token_count = gsl_tokenizer_encode_piece(
        &tokenizer, probe, sizeof(probe), tokens, sizeof(tokens) / sizeof(tokens[0])
    );
    if (token_count != 2u || tokens[0] != 1064u || tokens[1] != 6988u) {
        fputs("BPE encoding mismatch\n", stderr);
        return 1;
    }
    decoded_count = gsl_tokenizer_decode(
        &tokenizer, tokens, token_count, decoded, sizeof(decoded)
    );
    if (decoded_count != sizeof(probe) || memcmp(decoded, probe, sizeof(probe)) != 0) {
        fputs("BPE decoding mismatch\n", stderr);
        return 1;
    }
    gsl_tokenizer_close(&tokenizer);
    free(container);
    puts("tokenizer tests passed");
    return 0;
}
