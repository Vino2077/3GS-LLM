#include <stddef.h>

void *memcpy(void *destination, const void *source, size_t length)
{
    unsigned char *target = (unsigned char *)destination;
    const unsigned char *input = (const unsigned char *)source;
    size_t index;
    for (index = 0u; index < length; ++index) {
        target[index] = input[index];
    }
    return destination;
}

void *memmove(void *destination, const void *source, size_t length)
{
    unsigned char *target = (unsigned char *)destination;
    const unsigned char *input = (const unsigned char *)source;
    size_t index;
    if (target < input) {
        for (index = 0u; index < length; ++index) {
            target[index] = input[index];
        }
    } else if (target > input) {
        for (index = length; index > 0u; --index) {
            target[index - 1u] = input[index - 1u];
        }
    }
    return destination;
}

void *memset(void *destination, int value, size_t length)
{
    unsigned char *target = (unsigned char *)destination;
    size_t index;
    for (index = 0u; index < length; ++index) {
        target[index] = (unsigned char)value;
    }
    return destination;
}

int memcmp(const void *left, const void *right, size_t length)
{
    const unsigned char *a = (const unsigned char *)left;
    const unsigned char *b = (const unsigned char *)right;
    size_t index;
    for (index = 0u; index < length; ++index) {
        if (a[index] != b[index]) {
            return (int)a[index] - (int)b[index];
        }
    }
    return 0;
}

void memset_pattern16(void *destination, const void *pattern, size_t length)
{
    unsigned char *target = (unsigned char *)destination;
    const unsigned char *bytes = (const unsigned char *)pattern;
    size_t index;
    for (index = 0u; index < length; ++index) {
        target[index] = bytes[index & 15u];
    }
}
