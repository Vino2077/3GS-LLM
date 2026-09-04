#import <UIKit/UIKit.h>
#import "Sources/GSLAppDelegate.h"

int main(int argc, char *argv[])
{
    NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
    int result = UIApplicationMain(argc, argv, nil, @"GSLAppDelegate");
    [pool drain];
    return result;
}
