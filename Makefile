ARCHS := armv7
TARGET := iphone:clang:10.3:6.0
PACKAGE_FORMAT := ipa
INSTALL_TARGET_PROCESSES := ThreeGSLLM

include $(THEOS)/makefiles/common.mk

APPLICATION_NAME := ThreeGSLLM

ThreeGSLLM_FILES := \
	main.m \
	Sources/GSLAppDelegate.m \
	Sources/GSLChatViewController.m \
	Sources/GSLBenchmarkViewController.m \
	Sources/GSLTokenizerBridge.m \
	Runtime/int8_gemv.c \
	Runtime/model_format.c \
	Runtime/tokenizer.c \
	Runtime/decoder.c \
	Runtime/sampler.c \
	Runtime/benchmark.c
ThreeGSLLM_FRAMEWORKS := UIKit CoreGraphics Foundation
ThreeGSLLM_CFLAGS := -std=gnu11 -O3 -fno-objc-arc -fblocks -marm -mfpu=neon -Wall -Wextra
ThreeGSLLM_LDFLAGS := -Wl,-dead_strip

include $(THEOS_MAKE_PATH)/application.mk
