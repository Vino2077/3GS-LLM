#import "GSLChatViewController.h"

#import "GSLBenchmarkViewController.h"
#import "../Runtime/model_format.h"
#import "../Runtime/sampler.h"

#include <dispatch/dispatch.h>
#include <float.h>
#include <stdlib.h>
#include <time.h>

#define GSL_MAX_NEW_TOKENS 96u

@implementation GSLChatViewController

@synthesize conversationView = _conversationView;
@synthesize inputField = _inputField;
@synthesize sendButton = _sendButton;
@synthesize statusLabel = _statusLabel;

- (void)loadView
{
    CGRect screen = [[UIScreen mainScreen] applicationFrame];
    UIView *root = [[UIView alloc] initWithFrame:screen];
    CGFloat controlsY = screen.size.height - 76.0f;
    root.backgroundColor = [UIColor colorWithWhite:0.94f alpha:1.0f];
    root.autoresizingMask = UIViewAutoresizingFlexibleWidth |
                            UIViewAutoresizingFlexibleHeight;

    UITextView *conversation = [[UITextView alloc]
        initWithFrame:CGRectMake(8.0f, 8.0f, screen.size.width - 16.0f,
                                 controlsY - 36.0f)];
    conversation.autoresizingMask = UIViewAutoresizingFlexibleWidth |
                                     UIViewAutoresizingFlexibleHeight;
    conversation.editable = NO;
    conversation.font = [UIFont systemFontOfSize:14.0f];
    conversation.text = @"Полностью локальная 3GS-LM\n\n"
                         @"DTF-язык + локальный Stage 3 для более прямых ответов. "
                         @"Она может ошибаться и использовать грубую лексику.";
    [root addSubview:conversation];

    UILabel *status = [[UILabel alloc]
        initWithFrame:CGRectMake(10.0f, controlsY - 26.0f,
                                 screen.size.width - 20.0f, 20.0f)];
    status.autoresizingMask = UIViewAutoresizingFlexibleTopMargin |
                              UIViewAutoresizingFlexibleWidth;
    status.backgroundColor = [UIColor clearColor];
    status.font = [UIFont systemFontOfSize:11.0f];
    status.textColor = [UIColor darkGrayColor];
    status.text = @"Загрузка модели…";
    [root addSubview:status];

    UITextField *input = [[UITextField alloc]
        initWithFrame:CGRectMake(10.0f, controlsY,
                                 screen.size.width - 92.0f, 36.0f)];
    input.autoresizingMask = UIViewAutoresizingFlexibleTopMargin |
                             UIViewAutoresizingFlexibleWidth;
    input.borderStyle = UITextBorderStyleRoundedRect;
    input.placeholder = @"Сообщение";
    input.returnKeyType = UIReturnKeySend;
    input.delegate = self;
    input.enabled = NO;
    [root addSubview:input];

    UIButton *send = [UIButton buttonWithType:UIButtonTypeRoundedRect];
    send.frame = CGRectMake(screen.size.width - 76.0f, controlsY, 66.0f, 36.0f);
    send.autoresizingMask = UIViewAutoresizingFlexibleTopMargin |
                            UIViewAutoresizingFlexibleLeftMargin;
    [send setTitle:@"Ответ" forState:UIControlStateNormal];
    [send addTarget:self action:@selector(sendMessage:)
     forControlEvents:UIControlEventTouchUpInside];
    send.enabled = NO;
    [root addSubview:send];

    self.conversationView = conversation;
    self.inputField = input;
    self.sendButton = send;
    self.statusLabel = status;
    [conversation release];
    [input release];
    [status release];
    self.view = root;
    [root release];

    self.title = @"3GS-LLM";
    _samplerPreset = GSL_SAMPLER_PRESET_A;
    self.navigationItem.leftBarButtonItem = [[[UIBarButtonItem alloc]
        initWithTitle:@"Сэмпл: A"
                style:UIBarButtonItemStyleBordered
               target:self
               action:@selector(cycleSamplingPreset:)] autorelease];
    self.navigationItem.rightBarButtonItem = [[[UIBarButtonItem alloc]
        initWithTitle:@"Тест"
                style:UIBarButtonItemStyleBordered
               target:self
               action:@selector(showBenchmark:)] autorelease];
    [self loadModelInBackground];
}

- (void)cycleSamplingPreset:(id)sender
{
    (void)sender;
    _samplerPreset = (GSLSamplerPreset)(
        ((unsigned)_samplerPreset + 1u) % (unsigned)GSL_SAMPLER_PRESET_COUNT
    );
    self.navigationItem.leftBarButtonItem.title = [NSString stringWithFormat:
        @"Сэмпл: %s", gsl_sampler_preset_name(_samplerPreset)];
}

- (void)viewDidLoad
{
    [super viewDidLoad];
    [[NSNotificationCenter defaultCenter]
        addObserver:self selector:@selector(keyboardWillShow:)
              name:UIKeyboardWillShowNotification object:nil];
    [[NSNotificationCenter defaultCenter]
        addObserver:self selector:@selector(keyboardWillHide:)
              name:UIKeyboardWillHideNotification object:nil];
}

- (void)layoutForKeyboardHeight:(CGFloat)keyboardHeight
{
    CGRect bounds = self.view.bounds;
    CGFloat controlsY = bounds.size.height - 76.0f - keyboardHeight;
    if (controlsY < 76.0f) {
        controlsY = 76.0f;
    }
    self.conversationView.frame = CGRectMake(
        8.0f, 8.0f, bounds.size.width - 16.0f, controlsY - 36.0f
    );
    self.statusLabel.frame = CGRectMake(
        10.0f, controlsY - 26.0f, bounds.size.width - 20.0f, 20.0f
    );
    self.inputField.frame = CGRectMake(
        10.0f, controlsY, bounds.size.width - 92.0f, 36.0f
    );
    self.sendButton.frame = CGRectMake(
        bounds.size.width - 76.0f, controlsY, 66.0f, 36.0f
    );
}

- (void)viewDidLayoutSubviews
{
    [super viewDidLayoutSubviews];
    [self layoutForKeyboardHeight:_keyboardHeight];
}

- (void)keyboardWillShow:(NSNotification *)notification
{
    CGRect keyboard = [[[notification userInfo]
        objectForKey:UIKeyboardFrameEndUserInfoKey] CGRectValue];
    CGRect local = [self.view convertRect:keyboard fromView:nil];
    NSTimeInterval duration = [[notification.userInfo
        objectForKey:UIKeyboardAnimationDurationUserInfoKey] doubleValue];
    _keyboardHeight = self.view.bounds.size.height - CGRectGetMinY(local);
    [UIView animateWithDuration:duration animations:^{
        [self layoutForKeyboardHeight:_keyboardHeight];
    }];
}

- (void)keyboardWillHide:(NSNotification *)notification
{
    NSTimeInterval duration = [[notification.userInfo
        objectForKey:UIKeyboardAnimationDurationUserInfoKey] doubleValue];
    _keyboardHeight = 0.0f;
    [UIView animateWithDuration:duration animations:^{
        [self layoutForKeyboardHeight:0.0f];
    }];
}

- (void)showBenchmark:(id)sender
{
    (void)sender;
    GSLBenchmarkViewController *controller =
        [[GSLBenchmarkViewController alloc] init];
    [self.navigationController pushViewController:controller animated:YES];
    [controller release];
}

- (void)loadModelInBackground
{
    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
        NSError *error = nil;
        NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"model"
                                                              ofType:@"bin"];
        NSString *tokenizerPath = [[NSBundle mainBundle]
            pathForResource:@"tokenizer" ofType:@"bin"];
        BOOL loaded = NO;
        if (modelPath != nil && tokenizerPath != nil) {
            _modelData = [[NSData alloc] initWithContentsOfFile:modelPath
                                                        options:NSDataReadingMappedIfSafe
                                                          error:&error];
            _tokenizerBridge = [[GSLTokenizerBridge alloc]
                initWithPath:tokenizerPath error:&error];
            if (_modelData != nil && _tokenizerBridge != nil &&
                gsl_model_weights_open(&_modelWeights, [_modelData bytes],
                                       [_modelData length])) {
                _decoder = gsl_decoder_create(&_modelWeights);
                loaded = _decoder != NULL;
            }
        }
        [self performSelectorOnMainThread:@selector(modelDidLoad:)
                               withObject:[NSNumber numberWithBool:loaded]
                            waitUntilDone:NO];
        [pool drain];
    });
}

- (void)modelDidLoad:(NSNumber *)loaded
{
    if ([loaded boolValue]) {
        self.statusLabel.text = [NSString stringWithFormat:
            @"Готово • %.1f МБ RAM декодера",
            (double)gsl_decoder_memory_bytes() / (1024.0 * 1024.0)];
        self.inputField.enabled = YES;
        self.sendButton.enabled = YES;
        [self.inputField becomeFirstResponder];
    } else {
        self.statusLabel.text = @"Не удалось загрузить model.bin/tokenizer.bin";
    }
}

- (BOOL)textFieldShouldReturn:(UITextField *)textField
{
    if (textField == self.inputField && self.sendButton.enabled) {
        [self sendMessage:self.sendButton];
        return NO;
    }
    return YES;
}

- (void)sendMessage:(id)sender
{
    (void)sender;
    NSString *prompt = [self.inputField.text
        stringByTrimmingCharactersInSet:[NSCharacterSet whitespaceAndNewlineCharacterSet]];
    if ([prompt length] == 0u || _generating || _decoder == NULL) {
        return;
    }
    _generating = YES;
    [_activePrompt release];
    _activePrompt = [prompt copy];
    self.inputField.text = @"";
    self.inputField.enabled = NO;
    self.sendButton.enabled = NO;
    self.navigationItem.leftBarButtonItem.enabled = NO;
    [self.inputField resignFirstResponder];
    self.conversationView.text = [NSString stringWithFormat:
        @"Вы: %@\n\n3GS-LM: …", prompt];
    self.statusLabel.text = @"Читаю сообщение…";

    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        [self generateReplyForPrompt:prompt];
    });
}

- (void)generateReplyForPrompt:(NSString *)prompt
{
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    NSData *encoded = [_tokenizerBridge encodedTokensForString:prompt];
    const uint16_t *promptTokens = (const uint16_t *)[encoded bytes];
    size_t promptCount = [encoded length] / sizeof(uint16_t);
    const size_t promptLimit = GSL_CONTEXT_LENGTH - 3u - GSL_MAX_NEW_TOKENS;
    uint16_t history[GSL_CONTEXT_LENGTH];
    uint16_t response[GSL_MAX_NEW_TOKENS];
    float *logits = (float *)malloc(GSL_VOCAB_SIZE * sizeof(float));
    size_t historyCount = 0u;
    size_t responseCount = 0u;
    size_t index;
    GSLSampler sampler;
    BOOL success = encoded != nil && logits != NULL;

    if (promptCount > promptLimit) {
        promptTokens += promptCount - promptLimit;
        promptCount = promptLimit;
    }
    history[historyCount++] = 1u;
    history[historyCount++] = 3u;
    for (index = 0u; index < promptCount; ++index) {
        history[historyCount++] = promptTokens[index];
    }
    history[historyCount++] = 4u;
    gsl_decoder_reset(_decoder);
    for (index = 0u; success && index < historyCount; ++index) {
        success = gsl_decoder_step(_decoder, history[index], logits) != 0;
    }

    gsl_sampler_initialize(&sampler, (uint32_t)time(NULL));
    gsl_sampler_apply_preset(&sampler, _samplerPreset);
    for (index = 0u; success && index < GSL_MAX_NEW_TOKENS; ++index) {
        uint16_t token;
        logits[0] = -FLT_MAX;
        logits[1] = -FLT_MAX;
        logits[3] = -FLT_MAX;
        logits[4] = -FLT_MAX;
        logits[5] = -FLT_MAX;
        token = gsl_sampler_sample(&sampler, logits, GSL_VOCAB_SIZE,
                                   history, historyCount);
        if (token == 2u) {
            break;
        }
        response[responseCount++] = token;
        history[historyCount++] = token;
        {
            NSString *partial = [_tokenizerBridge stringForTokens:response
                                                             count:responseCount];
            if (partial != nil) {
                [self performSelectorOnMainThread:@selector(showPartialReply:)
                                       withObject:partial
                                    waitUntilDone:NO];
            }
        }
        if (index + 1u < GSL_MAX_NEW_TOKENS) {
            success = gsl_decoder_step(_decoder, token, logits) != 0;
        }
    }
    free(logits);
    [self performSelectorOnMainThread:@selector(generationDidFinish:)
                           withObject:[NSNumber numberWithBool:success]
                        waitUntilDone:NO];
    [pool drain];
}

- (void)showPartialReply:(NSString *)reply
{
    self.conversationView.text = [NSString stringWithFormat:
        @"Вы: %@\n\n3GS-LM: %@", _activePrompt, reply];
    NSRange bottom = NSMakeRange([self.conversationView.text length], 0u);
    [self.conversationView scrollRangeToVisible:bottom];
    self.statusLabel.text = @"Генерация локально…";
}

- (void)generationDidFinish:(NSNumber *)success
{
    _generating = NO;
    self.statusLabel.text = [success boolValue]
        ? @"Готово • полностью офлайн"
        : @"Ошибка генерации";
    self.inputField.enabled = YES;
    self.sendButton.enabled = YES;
    self.navigationItem.leftBarButtonItem.enabled = YES;
    [self.inputField becomeFirstResponder];
}

- (void)dealloc
{
    [[NSNotificationCenter defaultCenter] removeObserver:self];
    gsl_decoder_destroy(_decoder);
    [_tokenizerBridge release];
    [_modelData release];
    [_activePrompt release];
    [_conversationView release];
    [_inputField release];
    [_sendButton release];
    [_statusLabel release];
    [super dealloc];
}

@end
