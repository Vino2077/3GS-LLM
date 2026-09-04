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
    [button setTitle:@"Run model benchmark" forState:UIControlStateNormal];
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
    output.text = @"3GS-LLM candidate gate\n\n"
                   @"Candidate: 17.3M parameters, d=384, 8 layers, "
                   @"FFN=1024, vocabulary=8192. Tap to measure one full "
                   @"dense decoder step.";
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
            const size_t rows[] = { 384, 1024, 384, 8192 };
            const size_t cols[] = { 384, 384, 1024, 384 };
            const size_t iterations[] = { 100, 40, 40, 8 };
            double milliseconds[4] = { 0.0, 0.0, 0.0, 0.0 };
            int all_passed = 1;
            size_t i;

            [report appendString:@"Candidate 3GS-LM-17M\n"
                                  @"  d=384, layers=8, heads=6\n"
                                  @"  FFN=1024, vocab=8192\n"
                                  @"  context=256, INT8\n\n"];

            for (i = 0; i < 4; ++i) {
                GSLBenchmarkResult result;
                if (gsl_benchmark_gemv(rows[i], cols[i], iterations[i], &result)) {
                    milliseconds[i] = result.milliseconds_per_iteration;
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
                    all_passed = 0;
                    [report appendFormat:@"%lu x %lu: allocation failed\n\n",
                                          (unsigned long)rows[i],
                                          (unsigned long)cols[i]];
                }
            }

            if (all_passed) {
                double dense_step_ms =
                    8.0 * (4.0 * milliseconds[0] +
                           2.0 * milliseconds[1] +
                           milliseconds[2]) +
                    milliseconds[3];
                double kernel_ceiling = 1000.0 / dense_step_ms;
                [report appendFormat:@"Estimated dense decoder step:\n"
                                      @"  %.2f ms / token\n"
                                      @"  %.1f token/s kernel ceiling\n\n",
                                      dense_step_ms, kernel_ceiling];
            }

            [report appendString:@"Send the complete screen back. The final "
                                  @"runtime will be slower because attention, "
                                  @"normalization and sampling are not included."];
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
