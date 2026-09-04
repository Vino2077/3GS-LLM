# Training and quality evaluation

The model shape, tokenizer, context, and iPhone file formats remain frozen. This
pipeline improves prompt/response alignment without adding parameters or work to
the decoder.

The source DTF dataset and generated weights stay outside Git. Its Hugging Face
card does not declare a license, so redistribution requires a separate
provenance and permission review.

## What was wrong with the old response stage

Stage 1 correctly samples random windows from a packed token stream for language
model pretraining. The former Stage 2 reused the same sampler and merely masked
non-response tokens. A random window could therefore start inside a comment or
inside a reply, so the model often optimized reply tokens without seeing the
`USER -> ASSISTANT` boundary or the complete prompt.

`--response-only` now selects `AlignedPairStream`. Tokenization writes one
uint64 offset per pair, and every sample has this form:

```text
<BOS><USER>prompt<ASSISTANT>response<EOS><PAD>...
```

Padding and the user prompt have zero loss. Only response tokens and EOS are
optimized. Long pairs retain as much of the end of the prompt as possible and
the beginning of the response, reserving up to 64 response tokens. This matches
the iPhone's left-truncation of long prompts.

## Base and aligned DTF stages

Prepare the original corpus and tokenizer as before:

```powershell
python training/download_dtf.py C:\work\dtf-raw
python training/prepare_dtf.py C:\work\dtf-raw C:\work\dtf-clean
python training/train_tokenizer.py C:\work\dtf-clean\train.jsonl.gz C:\work\tokenizer
python training/tokenize_dataset.py C:\work\dtf-clean C:\work\tokenizer\tokenizer.json C:\work\dtf-tokens-aligned
```

Stage 1 still uses packed random windows:

```powershell
python training/train.py C:\work\dtf-tokens-aligned C:\work\checkpoints-pretrain `
  --steps 6000 --batch-size 64 --gradient-accumulation 4
```

Recommended new Stage 2 starts from the Stage 1 checkpoint, not from a legacy
random-window response checkpoint:

```powershell
python training/train.py C:\work\dtf-tokens-aligned C:\work\checkpoints-aligned `
  --steps 1200 --batch-size 64 --gradient-accumulation 4 `
  --warmup-steps 50 --learning-rate 3e-5 --min-learning-rate 3e-6 `
  --eval-interval 100 --eval-batches 50 --save-interval 100 `
  --response-only `
  --initial-weights C:\work\checkpoints-pretrain\checkpoint-06000.pt
```

Resume with the same schedule and `--resume checkpoint-NNNNN.pt`. Checkpoints
record whether their data mode was `packed_stream` or `aligned_pairs` and reject
an incompatible resume.

## Fixed prompt evaluation and sampling

`eval_prompts.json` contains 41 prompts across simple questions, reactions,
games, daily questions, DTF-style comments, identity, direct-answer tests, and
known failures. Seeds are stable across checkpoints and presets.

```powershell
python training/evaluate_quality.py `
  C:\work\checkpoints\checkpoint-00100.pt `
  C:\work\tokenizer\tokenizer.json `
  training\eval_prompts.json `
  C:\work\evaluation\aligned-sft-00100.txt `
  --presets legacy A B C near-greedy greedy `
  --seed 20260904 --max-new-tokens 48
```

For several checkpoints:

```powershell
python training/evaluate_quality_suite.py `
  C:\work\tokenizer\tokenizer.json training\eval_prompts.json `
  C:\work\evaluation `
  C:\work\checkpoints-pretrain\checkpoint-06000.pt `
  C:\work\checkpoints-aligned\checkpoint-01200.pt `
  C:\work\checkpoints-distilled\checkpoint-00400.pt `
  --presets C
```

Each report contains actual responses plus average response length, EOS rate,
repeated 3-gram rate, and a deliberately simple topic-term hit rate. The last
metric can detect gross failures but is not a semantic judge; read the responses
before selecting a checkpoint.

The initial legacy checkpoint produced these aggregate results:

| Preset | Topic-term hit | Repeated 3-gram | EOS |
|---|---:|---:|---:|
| legacy (0.8 / 40 / 1.08) | 26.8% | 1.67% | 70.7% |
| A (0.45 / 10 / 1.00) | 34.1% | 21.19% | 63.4% |
| B (0.25 / 5 / 1.00) | 43.9% | 21.54% | 68.3% |
| C (0.6 / 20 / 1.02) | 26.8% | 4.72% | 85.4% |
| near-greedy | 48.8% | 29.31% | 58.5% |
| greedy | 46.3% | 29.72% | 58.5% |

Low temperature appears more topical mostly because this checkpoint copies the
prompt and loops. Preset C is the safer temporary default: it sharply improves
completion and controls repetition. Re-run the same comparison after aligned
and distilled SFT; the best preset can change with the weights.

## FP, exported INT8, and C-runtime parity

Export weights, then compare forced decoder paths:

```powershell
python training/export_int8.py checkpoint.pt C:\work\model.bin
python training/compare_runtime.py checkpoint.pt tokenizer.json C:\work\model.bin `
  C:\work\runtime-comparison.json --steps 6 `
  --prompts "Скайрим играл?" "Ты кто?" "Киберпанк 2077 великая игра"
```

The comparison feeds FP-greedy token IDs to all backends, removing sampling
from the experiment. It records top-10 logits and errors for:

1. the FP32 PyTorch checkpoint;
2. simulated row-wise-weight/dynamic-activation INT8;
3. tensors parsed back from the exported `model.bin`.

`Tests/runtime_trace.c` runs the real `Runtime/decoder.c`. It can be built as a
native executable, or as WASI where Windows application control blocks unsigned
local executables. Pass it with `--native-executable`, or use
`--native-wasm trace.wasm --node node.exe`.

On the legacy checkpoint, simulated INT8 retained 95-98% of FP top-10 tokens.
The exported container matched the simulation exactly. The C decoder retained
95-100% of exported top-10 tokens and agreed on top-1 for 17 of 18 forced steps;
the one difference was a near-tie. This indicates that the observed semantic
failures originate in training, not corrupted bytes or a material runtime
divergence.

## Likes experiments

Inspect and build controlled subsets:

```powershell
python training/analyze_dtf_likes.py C:\work\dtf-clean
python training/prepare_quality_subset.py C:\work\dtf-clean C:\work\dtf-likes50 `
  --min-response-likes 50
python training/tokenize_dataset.py C:\work\dtf-likes50 tokenizer.json C:\work\dtf-likes50-tokens
```

Current subset sizes are 417,016 pairs for likes >= 10, 141,750 for >= 25, and
52,772 for >= 50. In equal 100-step pilots, full aligned DTF reached response
loss 3.87409 on the common validation set; likes >= 50 reached 3.88232 and did
not visibly improve the fixed prompts. Treat likes as one mixture signal, not a
replacement for semantic distillation. A future run can mix a minority of
high-like originals into distilled SFT.

## Distillation and quality SFT

Select 30k-100k unique real parent comments without copying their child reply:

```powershell
python training/build_distillation_prompts.py `
  C:\work\dtf-clean\train.jsonl.gz C:\work\distill-prompts.jsonl `
  --count 50000 --seed 20260904
python training/make_teacher_requests.py `
  C:\work\distill-prompts.jsonl training\teacher_prompt.txt `
  C:\work\teacher-requests.jsonl
```

The request format is provider-neutral: `id`, `system`, and `prompt`. Run it
through the chosen large teacher separately. The collected file must contain
one JSON object per line with `id`, `prompt`, and the teacher's `response`.

Validate, split by a stable prompt hash, exclude eval leakage, and tokenize:

```powershell
python training/import_teacher_outputs.py `
  C:\work\teacher-outputs.jsonl C:\work\distilled-sft.jsonl `
  --prompts C:\work\distill-prompts.jsonl
python training/prepare_sft.py `
  C:\work\distilled-sft.jsonl C:\work\distilled-clean `
  --exclude-prompts training\eval_prompts.json
python training/tokenize_dataset.py `
  C:\work\distilled-clean C:\work\tokenizer\tokenizer.json `
  C:\work\distilled-tokens
```

Recommended Stage 3 for about 50k pairs is deliberately short and low-rate:

```powershell
python training/train.py C:\work\distilled-tokens C:\work\checkpoints-distilled `
  --steps 300 --batch-size 64 --gradient-accumulation 4 `
  --warmup-steps 20 --learning-rate 1.5e-5 --min-learning-rate 1.5e-6 `
  --eval-interval 50 --eval-batches 50 --save-interval 50 `
  --response-only `
  --initial-weights C:\work\checkpoints-aligned\checkpoint-01200.pt
```

Evaluate every saved checkpoint by response loss and fixed generations. Stop
early if responses become generic, repetitive, or lose DTF tone even while
validation loss continues to improve.

## Smoke tests

```powershell
python training/smoke_test.py
python training/sampling_smoke_test.py
python training/pipeline_smoke_test.py
python training/sft_pipeline_smoke_test.py C:\work\tokenizer\tokenizer.json
python -m compileall -q training Scripts
```
