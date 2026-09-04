#import "GSLTokenizerBridge.h"

#include <stdlib.h>

@implementation GSLTokenizerBridge

- (id)initWithPath:(NSString *)path error:(NSError **)error
{
    self = [super init];
    if (self != nil) {
        _containerData = [[NSData alloc] initWithContentsOfFile:path
                                                        options:NSDataReadingMappedIfSafe
                                                          error:error];
        if (_containerData == nil ||
            !gsl_tokenizer_open(&_tokenizer, [_containerData bytes],
                                [_containerData length])) {
            [self release];
            return nil;
        }
        NSString *pattern =
            @"'s|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+";
        _piecePattern = [[NSRegularExpression alloc] initWithPattern:pattern
                                                             options:0
                                                               error:error];
        if (_piecePattern == nil) {
            [self release];
            return nil;
        }
    }
    return self;
}

- (const GSLTokenizer *)tokenizer
{
    return &_tokenizer;
}

- (NSData *)encodedTokensForString:(NSString *)string
{
    NSString *normalized = [string precomposedStringWithCanonicalMapping];
    NSArray *matches = [_piecePattern matchesInString:normalized
                                               options:0
                                                 range:NSMakeRange(0, [normalized length])];
    NSMutableData *result = [NSMutableData data];
    for (NSTextCheckingResult *match in matches) {
        NSString *piece = [normalized substringWithRange:[match range]];
        NSData *utf8 = [piece dataUsingEncoding:NSUTF8StringEncoding];
        NSUInteger byteCount = [utf8 length];
        uint16_t *tokens;
        size_t tokenCount;
        if (byteCount == 0u) {
            continue;
        }
        tokens = (uint16_t *)malloc(byteCount * sizeof(uint16_t));
        if (tokens == NULL) {
            return nil;
        }
        tokenCount = gsl_tokenizer_encode_piece(
            &_tokenizer, (const uint8_t *)[utf8 bytes], byteCount,
            tokens, byteCount
        );
        if (tokenCount == 0u) {
            free(tokens);
            return nil;
        }
        [result appendBytes:tokens length:tokenCount * sizeof(uint16_t)];
        free(tokens);
    }
    return result;
}

- (NSString *)stringForTokens:(const uint16_t *)tokens count:(NSUInteger)count
{
    size_t required = gsl_tokenizer_decode(
        &_tokenizer, tokens, count, NULL, 0u
    );
    uint8_t *bytes;
    NSString *result;
    if (required == 0u) {
        return @"";
    }
    bytes = (uint8_t *)malloc(required);
    if (bytes == NULL) {
        return nil;
    }
    if (gsl_tokenizer_decode(&_tokenizer, tokens, count, bytes, required) !=
        required) {
        free(bytes);
        return nil;
    }
    result = [[[NSString alloc] initWithBytes:bytes
                                       length:required
                                     encoding:NSUTF8StringEncoding] autorelease];
    free(bytes);
    return result;
}

- (void)dealloc
{
    gsl_tokenizer_close(&_tokenizer);
    [_piecePattern release];
    [_containerData release];
    [super dealloc];
}

@end
