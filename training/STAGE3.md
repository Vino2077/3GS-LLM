# Stage 3: semantic quality experiment

## v0.0.5 release result

The released recipe keeps the 17,308,032-parameter architecture unchanged.
Qwen3.5-27B Q3_K_M ran locally with no paid API. It correctly classified all
24 deliberately good/bad calibration decisions. From 400 real DTF prompts its
generation/judge pass accepted 129; a maximum-score gate retained 40 and fixed
manual review rejected 9 more, leaving 31 distinct teacher pairs. Repeated
sampling of those 31 is reported as training draws, not additional examples.

The selected `G` mixture contains 3,922 unique pairs and 5,893 draws:
3,200 game-alignment, 1,005 simple-alignment, 696 conversation-anchor,
372 reaction and 620 repeated draws from the 31 reviewed teacher pairs. Two
final pilots compared lower and higher teacher/general oversampling. `G` was
better on saved real answers. Full Stage 3 ran 600 steps at 2e-5 -> 5e-6,
batch 32, accumulation 2, with checkpoints every 100 steps. Checkpoint 400 was
selected by answer inspection; lower validation loss after it did not justify
the regressions seen in several replies.

The release sampler is preset A (temperature 0.45, top-k 10, repetition penalty
1.0). On 156 held-out development prompts it produced 0% prompt-copy, 0.16%
repeated 3-grams and 99.36% EOS for seed 42. On the 12 final prompts across
three seeds it produced 0% prompt-copy, 0.12% repeated 3-grams and 100% EOS.
These mechanical metrics are not claimed as semantic scores. Manual inspection
shows a useful improvement on several target prompts but also a clear 17M
capacity ceiling; the model is released as "good enough", not as solved.

INT8 response loss changed from 0.22978 to 0.23199 (+0.00221). Exported INT8
matches simulated INT8 exactly; forced-token C/WASI traces retain matching top-1
and top-10 on the release parity run. Private corpus rows and weights remain
outside Git.

The iPhone model architecture and C runtime are unchanged. A Stage 3 release
requires a semantic improvement on held-out prompts; lexical topic hits and
validation loss are insufficient.

## Current executed work (2026-09-05)

**Zero-cost continuation:** paid APIs are prohibited. The newer artifact ledger
is `../work/stage3-free/CURRENT.md`. Six actual student experiments A–F have now
run on 3,891 explicitly authored/template-expanded pairs (repeated mixture draws
are counted separately). None passed a small stratified semantic inspection.
Reduced loops and low loss alone do not authorize release. Qwen3-14B batched
teacher/judge also failed manual inspection. Its accepted export is provisional,
not training-ready. Qwen3.5-27B Q3_K_M is the next local calibration attempt.
The older notes below describe earlier engineering and teacher pilots.

- Scanned 709,499 cleaned training rows. The inexpensive filter found 535,971
  unique eligible parents and sampled 60,000 deterministically. **Eligibility
  does not certify self-containedness.** 308,933 original response candidates
  passed weak length/repetition/likes >=10 checks; these still need semantic review.
- Froze 156 development prompts and the user's separate 12 final prompts.
  Exclude both from Stage 3 data. These are held out from Stage 3, not a proven
  never-seen subset of all Stage 1/2 data. Near-duplicate exclusion is lexical
  and cannot guarantee paraphrase isolation.
- Tested the installed Qwen2.5-Coder-14B Q4 teacher/judge. It rejected good answers
  and rated obvious nonsense as highly relevant. Do not trust it for selection.
- Downloaded Qwen3.5-9B Q6_K and ran an 18-answer judge calibration. It passed
  those binary decisions, but manual review of real DTF generations found
  substantive mistakes among accepted answers. Calibration alone is insufficient.
- Completed a 100-prompt real-data teacher pilot: 56 locally accepted,
  13 rejected for context, 9 by the judge, 20 malformed generation outputs,
  and 2 malformed judgments. Manual review identified at least 8 clear
  substantive failures among the first 20 accepted outputs. This does not
  certify the other 12 or estimate quality across the entire source corpus.
- Real-data pilot outputs live outside Git under
  `../work/stage3/teacher-pilot-100`. Check its `stats.json` and durable
  `decisions.jsonl` for completion. This is **teacher generation**, not a student
  training pilot. No semantic success or Stage 3 model is claimed.
- Prefix-loss masking/gradient tests passed. A two-step CUDA smoke training
  and a restart from step 1 produced bit-identical step-2 model parameters.
  These are engineering checks, not quality-training results.

## Local teacher pilot

```powershell
python training/build_stage3_eval.py training
python training/filter_stage3.py ../work/dtf-clean/train.jsonl.gz ../work/stage3/filtered --exclude training/eval_stage3_exclude.json --limit 60000
python training/calibrate_teacher.py ../work/stage3/calibration --model 3gs-teacher-qwen35 --lmstudio-native
python training/distill_stage3.py ../work/stage3/filtered/teacher_prompts.jsonl ../work/stage3/teacher-pilot-100 --limit 100 --model 3gs-teacher-qwen35 --lmstudio-native
```

Local LM Studio listens on localhost:1234. Native transport disables reasoning
with the documented `reasoning: off` setting and does not store chats. That
endpoint does not expose a documented seed parameter: outputs are persisted,
but identical seed values do not make fresh teacher generations reproducible.
Student evaluation uses explicit PyTorch seeds.

Two candidates are generated without the original child response. A separate
call scores them in shuffled order. Accept only relevance/directness/fluency
>=4, style >=3, no repetition, no copied prompt and no invented hidden context.
Invalid JSON and bad schemas are rejected. All accepted and rejected decisions
are retained with model output and timing. The current teacher and judge are
the same local weights with different prompts; this is not an independent
model audit, and their correlated errors were observed in practice.

`accepted.jsonl` is a provisional export, **not an approved training corpus**.
Resume into the same directory only with the same input and prompt manifest.
Full JSONL rows are flushed and fsynced; a damaged partial last row after power
loss must be reviewed/recovered before resuming (the reader fails rather than
silently skipping corruption). No automatic startup or crash restart is installed.

## Prefix-loss experiment

`train.py --response-only --response-prefix-tokens 16 --response-prefix-weight 1.5`
weights the first 16 response tokens 1.5x. Prompt, padding and EOS are not boosted;
EOS retains weight 1. Validation uses ordinary response loss so weighted and
unweighted runs remain comparable. Default weight 1 preserves existing training.
Checkpoints record weighting, training arguments and RNG states; resume refuses
different weighting. Learning-rate schedules must retain the original final
step count for an exact continuation.

After a dataset passes human audit, compare equal-budget A (ordinary), B (prefix
weight), C (minority original DTF), and D (simple-to-mixed curriculum) runs from
the SAME aligned-v2 checkpoint with learning rate 1e-5..3e-5. Dataset creation,
mixture experiments and quality checkpoint selection are pending teacher quality;
no recipe has yet won.

## Evaluation and tests

```powershell
python training/evaluate_stage3.py ../work/checkpoints-aligned-v2/checkpoint-01200.pt ../work/dtf-tokenizer/tokenizer.json training/eval_stage3_dev.json ../work/stage3/eval-aligned-v2 --presets A B C near-greedy greedy legacy --seeds 20260905 42 789
python training/stage3_smoke_test.py
python training/sft_pipeline_smoke_test.py ../work/dtf-tokenizer/tokenizer.json
```

Evaluation writes a hash-checked manifest, individual answers with seeds/token
IDs, a readable report and copy/repetition/EOS/length metrics. Greedy runs once
per prompt because repeated seeds do not create independent evidence. Semantic
scores must come from a calibrated reviewer and human inspection; the evaluator
does not disguise word overlap as semantic relevance.

Only after verified improvement: export INT8, repeat forced-token runtime parity,
package the selected model and sampler, and write the IPA **only to the Desktop**.
Preserve aligned-v2 for rollback. Do not publish corpus rows or checkpoints to Git.
