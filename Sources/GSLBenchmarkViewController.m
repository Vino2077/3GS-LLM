#import "GSLBenchmarkViewController.h"
#import "../Runtime/benchmark.h"

#include <dispatch/dispatch.h>

@implementation GSLBenchmarkViewController

@synthesize runButton = _runButton;
@synthesize outputView = _outputView;

- (void)loadView
{
    CGRect screen = [[UIScreen mainScreen] applicationFrame];
    UIView *root = [[UIView alloc] initWithFrame:screen];
    root.backgroundColor = [UIColor colorWithWhite:0.94f alpha:1.0f];
    root.autoresizingMask = UIViewAutoresizingFlexibleWidth |
                            UIViewAutoresizingFlexibleHeight;

    UIButton *button = [UIButton buttonWithType:UIButtonTypeRoundedRect];
    button.frame = CGRectMake(20.0f, 14.0f, screen.size.width - 40.0f, 44.0f);
    button.autoresizingMask = UIViewAutoresizingFlexibleWidth;
    [button setTitle:@"Run INT8 benchmark" forState:UIControlStateNormal];
    [button addTarget:self
               action:@selector(runBenchmark:)
     forControlEvents:UIControlEventTouchUpInside];
    [root addSubview:button];

    UITextView *output = [[UITextView alloc]
        initWithFrame:CGRectMake(10.0f, 68.0f,
                                 screen.size.width - 20.0f,
                                 screen.size.height - 78.0f)];
    output.autoresizingMask = UIViewAutoresizingFlexibleWidth |
                              UIViewAutoresizingFlexibleHeight;
    output.editable = NO;
    output.font = [UIFont fontWithName:@"Courier" size:11.0f];
    output.text = @"3GS-LLM hardware gate\n\n"
                   @"This measures the INT8 matrix-vector operations needed "
                   @"for local token generation. Tap the button to begin.";
    [root addSubview:output];

    self.runButton = button;
    self.outputView = output;
    [output release];
    self.view = root;
    [root release];
}

- (void)runBenchmark:(id)sender
{
    (void)sender;
    self.runButton.enabled = NO;
    self.outputView.text = @"Verifying kernel and running benchmark...";

    dispatch_async(dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^{
        NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
        NSMutableString *report = [[NSMutableString alloc] init];

        int verified = gsl_verify_int8_kernel();
        [report appendFormat:@"Kernel: %@\n",
                             gsl_int8_neon_enabled() ? @"ARMv7 NEON" : @"scalar"];
        [report appendFormat:@"Verification: %@\n\n",
                             verified ? @"PASS" : @"FAIL"];

        if (verified) {
            const size_t rows[] = { 192, 512, 4096 };
            const size_t iterations[] = { 200, 100, 20 };
            size_t i;

            for (i = 0; i < 3; ++i) {
                GSLBenchmarkResult result;
                if (gsl_benchmark_gemv(rows[i], 192, iterations[i], &result)) {
                    [report appendFormat:@"%lu x %lu\n"
                                          @"  %.3f ms / GEMV\n"
                                          @"  %.2f MMAC/s\n"
                                          @"  checksum: %ld\n\n",
                                          (unsigned long)result.rows,
                                          (unsigned long)result.cols,
                                          result.milliseconds_per_iteration,
                                          result.mmac_per_second,
                                          (long)result.checksum];
                } else {
                    [report appendFormat:@"%lu x 192: allocation failed\n\n",
                                          (unsigned long)rows[i]];
                }
            }

            [report appendString:@"Send these numbers back to the project. "
                                  @"They determine the first model shape."];
        } else {
            [report appendString:@"The optimized kernel disagrees with the "
                                  @"reference implementation. Results discarded."];
        }

        [self performSelectorOnMainThread:@selector(showBenchmarkReport:)
                               withObject:report
                            waitUntilDone:NO];
        [report release];
        [pool drain];
    });
}

- (void)showBenchmarkReport:(NSString *)report
{
    self.outputView.text = report;
    self.runButton.enabled = YES;
}

- (void)dealloc
{
    [_runButton release];
    [_outputView release];
    [super dealloc];
}

@end
