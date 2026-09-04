#import "GSLAppDelegate.h"
#import "GSLChatViewController.h"

@implementation GSLAppDelegate

@synthesize window = _window;
@synthesize navigationController = _navigationController;

- (BOOL)application:(UIApplication *)application
        didFinishLaunchingWithOptions:(NSDictionary *)launchOptions
{
    (void)application;
    (void)launchOptions;

    GSLChatViewController *controller =
        [[GSLChatViewController alloc] init];
    UINavigationController *navigation =
        [[UINavigationController alloc] initWithRootViewController:controller];
    [controller release];

    UIWindow *window = [[UIWindow alloc] initWithFrame:[[UIScreen mainScreen] bounds]];
    window.rootViewController = navigation;
    [window makeKeyAndVisible];

    self.navigationController = navigation;
    self.window = window;
    [navigation release];
    [window release];
    return YES;
}

- (void)dealloc
{
    [_navigationController release];
    [_window release];
    [super dealloc];
}

@end
