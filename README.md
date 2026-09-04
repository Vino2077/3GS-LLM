# 3GS-LLM

A tiny, fully offline text-generation runtime for the iPhone 3GS running
iOS 6.1.6.

The project is deliberately starting with a hardware benchmark. Before a model
architecture is frozen, the real phone must tell us how quickly its 600 MHz
Cortex-A8 can execute the three INT8 matrix-vector shapes that will dominate
token generation.

## Current milestone: ARMv7/NEON benchmark

The first IPA contains a small UIKit application that measures:

- `192 x 192` -- attention projection;
- `512 x 192` -- feed-forward projection;
- `4096 x 192` -- vocabulary projection.

The app verifies the optimized dot product against a scalar implementation
before reporting milliseconds per matrix-vector multiply and MMAC/s. The
benchmark runs entirely on the CPU; no network connection or model file is
required.

## Build target

- Device: iPhone 3GS
- OS: iOS 6.1.6 (deployment target 6.0)
- Architecture: ARMv7 with NEON
- Language: C99 plus Objective-C with manual reference counting
- Build system: Theos on Linux
- Package: ad-hoc-signed IPA for a jailbroken device

## Downloading a build

Every push to `main` starts the **Build iOS 6 IPA** GitHub Actions workflow.
Open the workflow run, download the `3GS-LLM-IPA` artifact, and extract the IPA.

## Local build

With Theos and the iPhoneOS 10.3 SDK installed:

```sh
export THEOS="$HOME/theos"
make clean package FINALPACKAGE=1
```

The package is written to `packages/`.

## Planned path

1. Measure the real 3GS and record a performance baseline.
2. Implement the model file format, tokenizer, sampling, and KV cache.
3. Train a deliberately tiny Russian/English model on a modern computer.
4. Quantize weights to INT8 and validate output against the reference runtime.
5. Integrate streaming local generation into the iOS 6 interface.
