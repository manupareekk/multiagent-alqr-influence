# TruthfulQA MC1 (exact letter) + gemma-2-2b-it

Preferred eval: **multiple-choice**, score = extracted letter == gold letter.

## What to do (in order)

### 0) Laptop
Re-upload `~/Downloads/lqr-ma.zip` into Kaggle dataset **`lqr-maaaa`**.

### 1) Kaggle — MC baseline (no pickles, ~20–40 min)

Do **not** start pickle build until this looks good (B answers are letters like `B` / `C`).

```python
import os, shutil, zipfile
from pathlib import Path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

!pip uninstall -y torch torchvision torchaudio -q
!pip install -q torch --index-url https://download.pytorch.org/whl/cu124
!pip install -q transformers accelerate sentencepiece datasets

import torch
print(torch.__version__, torch.cuda.get_device_name(0))

%cd /kaggle/working
src = None
for p in Path("/kaggle/input").rglob("*"):
    if p.is_dir() and p.name.lower() in ("lqr-maaaa", "lqr-maaa", "lqr-maa", "lqr-ma"):
        src = p
        break
assert src, "attach lqr-maaaa"

dst = Path("/kaggle/working/lqr-activation-steering")
if dst.exists():
    shutil.rmtree(dst)
if (src / "multiagent").exists():
    shutil.copytree(src, dst)
elif (src / "lqr-activation-steering" / "multiagent").exists():
    shutil.copytree(src / "lqr-activation-steering", dst)
else:
    z = next(src.rglob("*.zip"))
    with zipfile.ZipFile(z) as zf:
        zf.extractall("/kaggle/working")
    if not dst.exists():
        cand = next(Path("/kaggle/working").rglob("multiagent")).parent
        shutil.move(str(cand), str(dst))

%cd /kaggle/working/lqr-activation-steering
assert Path("multiagent/tqa_scenarios.py").exists()

from huggingface_hub import login
login(token="HF_TOKEN_HERE")

!python -m multiagent.run_tqa_baseline \
  --hf-model google/gemma-2-2b-it \
  --scenario-set tqa_mc \
  --n-resist 10 --n-accept 10 --seed 0
```

**Pass gate**
- `B alone` / `B after` look like **`A`** / **`B`** / **`C`** (or short “Answer: B”), not essays
- Notes say `MC: ...`
- Paste SUCCESS line + 1 sample

### 2) After MC baseline looks good — rebuild pickles (hours)

```python
%cd /kaggle/working/lqr-activation-steering
!mkdir -p /kaggle/working/pickles_it_tqa_mc
!python -m multiagent.influence_data \
  --model gemma2bit \
  --prompt-source tqa \
  --pickle-dir /kaggle/working/pickles_it_tqa_mc \
  --n-jac 30
!cd /kaggle/working && zip -r influence-pickles-gemma2bit-tqa-mc.zip pickles_it_tqa_mc
```

### 3) λ sweep (after pickles)

```python
!python -m multiagent.run_lambda_sweep \
  --baseline alqr --steered alqr \
  --hf-model google/gemma-2-2b-it --model-key gemma-2-2b-it \
  --pickle-dir /kaggle/working/pickles_it_tqa_mc --no-quant \
  --scenario-set tqa_mc \
  --tqa-n-resist 10 --tqa-n-accept 10 --tqa-seed 0 \
  --lambdas 0.5,1.0,1.25,1.5,2.0
```
