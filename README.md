# 3GS-LLM

A tiny, fully offline text-generation runtime for the iPhone 3GS running
iOS 6.1.6.

The project is deliberately starting with a hardware benchmark. Before a model
architecture is frozen, the real phone must tell us how quickly its 600 MHz
Cortex-A8 can execute the three INT8 matrix-vector shapes that will dominate
token generation.

## Current milestone: 17M-parameter candidate benchmark

The first hardware run measured 269--416 MMAC/s on a real iPhone 3GS. That is
fast enough to test a substantially more useful model than the original
2M-parameter concept. The current IPA measures the exact dense projections for
the proposed `3GS-LM-17M` shape:

- model width 384, 8 transformer layers, 6 attention heads;
- SwiGLU feed-forward width 1024;
- 8192-token tied vocabulary;
- 256-token context;
- approximately 17.3M parameters and 17.5 MB of quantized weights.

The app verifies the optimized dot product against a scalar implementation,
reports every candidate matrix shape, and estimates the dense kernel time per
generated token. Attention, normalization, tokenizer and sampling overhead are
intentionally not included yet.

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
