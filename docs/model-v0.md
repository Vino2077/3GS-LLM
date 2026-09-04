# 3GS-LM-17M candidate

This shape is the first serious model candidate. It is based on the initial
real-device result of 269--416 MMAC/s and targets useful short-form dialogue at
roughly 3--5 words per second after full runtime overhead.

| Property | Value |
| --- | ---: |
| Decoder layers | 8 |
| Model width | 384 |
| Attention heads | 6 |
| Head dimension | 64 |
| Feed-forward width | 1024 |
| Activation | SwiGLU |
| Vocabulary | 8192 byte-level BPE tokens |
| Context | 256 tokens |
| Positional encoding | RoPE |
| Normalization | RMSNorm |
| Weight format | Per-row symmetric INT8 |
| KV cache target | INT8 with per-token scales |
| Tied token embedding / LM head | Yes |

## Parameter estimate

- tied embedding and LM head: `8192 * 384 = 3,145,728`;
- attention per layer: `4 * 384 * 384 = 589,824`;
- SwiGLU per layer: `3 * 384 * 1024 = 1,179,648`;
- two RMSNorm vectors per layer: `768`;
- eight layers plus final norm: approximately `17,308,032` parameters.

The quantized model should occupy about 17.5 MB including per-row scales. An
INT8 KV cache at 256 tokens costs roughly 1.6 MB. Both leave ample room for the
iOS process, tokenizer, scratch buffers, and the interface within the device's
256 MB physical memory.

## Performance gate

The candidate remains provisional until the second IPA measures these shapes
on the phone:

- four `384 x 384` attention projections per layer;
- two `1024 x 384` gate/up projections per layer;
- one `384 x 1024` down projection per layer;
- one `8192 x 384` vocabulary projection per generated token.

The displayed dense-step estimate is an upper bound. The final token rate must
also pay for attention over the KV cache, dynamic activation quantization,
RMSNorm, RoPE, residual operations, softmax, and sampling.
