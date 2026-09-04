# 3GS-LLM

A tiny, fully offline text-generation runtime for the iPhone 3GS running
iOS 6.1.6.

The architecture was selected from measurements on the real phone. The app is
now a local chat UI backed by a C99 transformer decoder; the hardware benchmark
remains available from the navigation bar.

## Current milestone: first trained offline build

The first hardware run measured 269--416 MMAC/s on a real iPhone 3GS. That is
fast enough to test a substantially more useful model than the original
2M-parameter concept. The frozen `3GS-LM-17M` shape is:

- model width 384, 8 transformer layers, 6 attention heads;
- SwiGLU feed-forward width 1024;
- 8192-token tied vocabulary;
- 256-token context;
- approximately 17.3M parameters and 17.5 MB of quantized weights.

The completed training pipeline cleans and tokenizes the
[DTF Comments Responses Counts](https://huggingface.co/datasets/SubMaroon/DTF_Comments_Responses_Counts)
dataset into 709,499 train pairs and 6,604 validation pairs. The native runtime
implements mapped binary model loading, byte-level BPE, a float KV cache,
RMSNorm, RoPE, causal attention, SwiGLU, top-k sampling, and row-wise INT8 NEON
matrix-vector products.

The first model was trained for 6,000 all-token steps followed by 600
response-only steps. On a fixed validation sample, the selected checkpoint has
a response loss of 3.91833. Simulating the exported row-wise weight and dynamic
activation INT8 path changes that loss by only +0.00175.

The dataset card does not declare a license. Source rows and trained weights are
therefore kept out of this public repository. A CI build is an asset-free app
shell; final personal-test IPAs are assembled locally by injecting the validated
model and tokenizer containers after the executable is built.

## Build target

- Device: iPhone 3GS
- OS: iOS 6.1.6 (deployment target 6.0)
- Architecture: ARMv7 with NEON
- Language: C99 plus Objective-C with manual reference counting
- Build system: Theos on Linux
- Package: ad-hoc-signed IPA for a jailbroken device

## Downloading a build

Every push to `main` starts the **Build iOS 6 IPA** GitHub Actions workflow.
The `3GS-LLM-IPA` artifact intentionally does not contain trained weights.

After exporting private assets, assemble a personal-test IPA with:

```sh
python Scripts/inject_model_assets.py shell.ipa model.bin tokenizer.bin final.ipa
python Scripts/verify_ipa.py final.ipa
```

The injection script validates both binary containers and refuses an IPA whose
resources are sealed by the executable signature.

## Local build

With Theos and the iPhoneOS 10.3 SDK installed:

```sh
export THEOS="$HOME/theos"
make clean package FINALPACKAGE=1
```

The package is written to `packages/`.

## Project path

1. Measure the real 3GS and record a performance baseline. Done.
2. Implement the model file format, tokenizer, sampling, and KV cache. Done.
3. Train the 17.3M-parameter model on the DTF reply corpus. Done.
4. Quantize weights to INT8 and validate output against the reference runtime. Done.
5. Inject the assets into the iOS 6 chat shell. Done.
6. Test generation speed, stability, and replies on the real device.
