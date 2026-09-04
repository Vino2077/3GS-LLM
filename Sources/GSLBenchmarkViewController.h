#import <UIKit/UIKit.h>

@interface GSLBenchmarkViewController : UIViewController {
    UIButton *_runButton;
    UITextView *_outputView;
}

@property(nonatomic, retain) UIButton *runButton;
@property(nonatomic, retain) UITextView *outputView;

@end
