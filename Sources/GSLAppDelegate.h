#import <UIKit/UIKit.h>

@interface GSLAppDelegate : UIResponder <UIApplicationDelegate> {
    UIWindow *_window;
    UINavigationController *_navigationController;
}

@property(nonatomic, retain) UIWindow *window;
@property(nonatomic, retain) UINavigationController *navigationController;

@end
