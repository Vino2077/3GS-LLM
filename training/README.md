# Training pipeline

The scripts intentionally keep the 4.29 GB source dataset and all generated
artifacts outside Git. The Hugging Face dataset card does not declare a license,
so neither the source rows nor trained weights should be redistributed without
separate permission and provenance review.

## Environment

Use Python 3.10 or newer with a CUDA build of PyTorch appropriate for the local
GPU, then install the remaining packages from `requirements.txt`.

## Pipeline

```powershell
python training/download_dtf.py C:\path\to\dtf-raw
python training/prepare_dtf.py C:\path\to\dtf-raw C:\path\to\dtf-clean
python training/train_tokenizer.py C:\path\to\dtf-clean\train.jsonl.gz C:\path\to\tokenizer
python training/tokenize_dataset.py C:\path\to\dtf-clean C:\path\to\tokenizer\tokenizer.json C:\path\to\tokens
python training/train.py C:\path\to\tokens C:\path\to\checkpoints
python training/train.py C:\path\to\tokens C:\path\to\response-checkpoints `
  --steps 600 --warmup-steps 30 --learning-rate 6e-5 --min-learning-rate 6e-6 `
  --response-only --initial-weights C:\path\to\checkpoints\checkpoint-06000.pt
```

An interrupted stage can be continued with `--resume checkpoint-NNNNN.pt` and
the same final `--steps` and schedule arguments. This restores both model and
AdamW optimizer state.

The cleaning split is grouped by a stable hash of the post title, so replies
from the same DTF post do not leak randomly between training and validation.
Deleted-comment markers, exact child-ID duplicates, echoes, and sequences too
long for useful 256-token packing are removed. Toxic comments are retained and
counted because the requested target is explicitly a DTF-domain model.

The first training stage predicts every corpus token. The shorter second stage
starts from that checkpoint and applies loss only to reply and end-of-reply
tokens, which adapts the language model to its chat role.

The tokenizer uses the standard byte-level pre-tokenizer split before BPE.
This keeps the vocabulary byte-complete while avoiding pathological training
time on hundreds of thousands of whole-comment strings. The same split must be
reproduced by the iOS runtime before applying the exported BPE merges.
