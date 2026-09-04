#import <Foundation/Foundation.h>

#import "../Runtime/tokenizer.h"

@interface GSLTokenizerBridge : NSObject {
    NSData *_containerData;
    NSRegularExpression *_piecePattern;
    GSLTokenizer _tokenizer;
}

@property(nonatomic, readonly) const GSLTokenizer *tokenizer;

- (id)initWithPath:(NSString *)path error:(NSError **)error;
- (NSData *)encodedTokensForString:(NSString *)string;
- (NSString *)stringForTokens:(const uint16_t *)tokens count:(NSUInteger)count;

@end
