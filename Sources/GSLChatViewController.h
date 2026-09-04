#import <UIKit/UIKit.h>

#import "GSLTokenizerBridge.h"
#import "../Runtime/decoder.h"
#import "../Runtime/sampler.h"

@interface GSLChatViewController : UIViewController <UITextFieldDelegate> {
    UITextView *_conversationView;
    UITextField *_inputField;
    UIButton *_sendButton;
    UILabel *_statusLabel;
    NSData *_modelData;
    GSLTokenizerBridge *_tokenizerBridge;
    GSLModelWeights _modelWeights;
    GSLDecoder *_decoder;
    BOOL _generating;
    NSString *_activePrompt;
    CGFloat _keyboardHeight;
    GSLSamplerPreset _samplerPreset;
}

@property(nonatomic, retain) UITextView *conversationView;
@property(nonatomic, retain) UITextField *inputField;
@property(nonatomic, retain) UIButton *sendButton;
@property(nonatomic, retain) UILabel *statusLabel;

@end
