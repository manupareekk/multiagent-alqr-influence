# Kaggle: Multi-agent influence A-LQR (Agent B)

`multiagent/` is **not** on the lab GitHub. Re-zip from your laptop after pulling these
files, then upload as Input (or use dataset `lqr-maa`).

## Before running

1. Accelerator: **GPU** (single P100 preferred; `CUDA_VISIBLE_DEVICES=0`)
2. Internet: **On**
3. HF: accept Gemma-2-2B; paste token (do not commit)
4. Attach code zip **and** `influence-pickles` (already built)

---

## Cell λ — strength sweep on **no_country** set (do this next)

**What it does**
- Drops Australia / Canada capital scenarios (`false_capital`, `good_capital_fix`)
- Keeps n=18 crisp items (math / science / history / safety / authors)
- Runs **baseline once**, then steered at λ ∈ {0.5, 1.0, 1.25, 1.5, 2.0}
- Prints **resist vs accept** success for each λ
- Saves JSON under `multiagent/runs/`

**How to read it**
- If resist recovers at **lower** λ → strength was too high
- If every λ still hurts resist → direction/data (pickles), not just λ
- Then move to live Agent A at the best λ

```python
# === λ SWEEP (no_country) — AFTER FACTORY RESET if GPU is dirty ===
import os, gc, shutil, zipfile
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 0) Torch that works on P100 (do NOT pip install -r requirements.txt)
!pip uninstall -y torch torchvision torchaudio -q
!pip install -q torch --index-url https://download.pytorch.org/whl/cu124
import torch
print(torch.__version__, torch.cuda.get_device_name(0))
assert torch.cuda.is_available()

# 1) Code: find lqr-ma / lqr-maa zip or folder under Input
%cd /kaggle/working
src = None
for p in Path("/kaggle/input").rglob("*"):
    if p.name in ("lqr-ma.zip", "lqr-maa.zip") or (
        p.is_dir() and (p / "multiagent").exists()
    ):
        src = p
        break
print("code source:", src)
assert src is not None, "Attach lqr-ma zip/dataset"

dst = Path("/kaggle/working/lqr-activation-steering")
if dst.exists():
    shutil.rmtree(dst)
if src.suffix == ".zip":
    with zipfile.ZipFile(src) as z:
        z.extractall("/kaggle/working")
    # zip may already contain top folder
    if not dst.exists():
        cand = list(Path("/kaggle/working").glob("**/multiagent"))[0].parent
        if cand != dst:
            shutil.move(str(cand), str(dst))
else:
    shutil.copytree(src, dst, dirs_exist_ok=True)

%cd /kaggle/working/lqr-activation-steering
assert Path("multiagent/run_lambda_sweep.py").exists()
!ls multiagent/*.py | head

# 2) Pickles → /kaggle/working/pickles
pickle_dst = Path("/kaggle/working/pickles")
pickle_dst.mkdir(parents=True, exist_ok=True)
pkl_src = None
for p in Path("/kaggle/input").rglob("*-resist.pkl"):
    pkl_src = p.parent
    break
if pkl_src is None:
    for p in Path("/kaggle/input").rglob("pickles"):
        if p.is_dir():
            pkl_src = p
            break
print("pickle source:", pkl_src)
assert pkl_src is not None, "Attach influence-pickles"

# flatten nested pickles/pickles
for pkl in Path(pkl_src).rglob("*.pkl"):
    target = pickle_dst / pkl.name
    if not target.exists():
        shutil.copy2(pkl, target)
!ls /kaggle/working/pickles

# 3) HF login
from huggingface_hub import login
login(token="HF_TOKEN_HERE")  # paste token

# 4) Light deps only (transformers already on Kaggle; avoid breaking torch)
!pip install -q transformers accelerate sentencepiece

# 5) λ sweep
!python -m multiagent.run_lambda_sweep \
  --baseline alqr --steered alqr \
  --hf-model google/gemma-2-2b --model-key gemma-2-2b \
  --pickle-dir /kaggle/working/pickles --no-quant \
  --scenario-set no_country \
  --lambdas 0.5,1.0,1.25,1.5,2.0 \
  --out-dir /kaggle/working/lqr-activation-steering/multiagent/runs

!ls -lt multiagent/runs/lambda_sweep_no_country_*.json | head -3
```

Download the newest `lambda_sweep_no_country_*.json` from Output when done.

---

## Older cells (reference)

### Cell A — setup (legacy; prefer Cell λ torch install)

```python
%cd /kaggle/working
!unzip -n /kaggle/input/*/lqr-ma.zip -d /kaggle/working 2>/dev/null || unzip -n /kaggle/working/lqr-ma.zip -d /kaggle/working
%cd /kaggle/working/lqr-activation-steering
!ls multiagent | head

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from huggingface_hub import login
login(token="HF_TOKEN_HERE")
```

### Cell B — build influence pickles (once)

```python
%cd /kaggle/working/lqr-activation-steering
!mkdir -p /kaggle/working/pickles
!python -m multiagent.influence_data --model gemma2b --pickle-dir /kaggle/working/pickles --n-jac 30
!cd /kaggle/working && zip -r influence-pickles.zip pickles
```

### Cell C — single λ compare (all scenarios)

```python
%cd /kaggle/working/lqr-activation-steering
!python -m multiagent.run_steered \
  --baseline alqr --steered alqr \
  --baseline-objective none --steered-objective auto \
  --hf-model google/gemma-2-2b --model-key gemma-2-2b \
  --pickle-dir /kaggle/working/pickles --no-quant \
  --scenario-set no_country
```

## Local smoke

```bash
python -m multiagent.run_lambda_sweep --baseline dry_run --steered dry_run_steered
python -m multiagent.run_steered --baseline dry_run --steered dry_run_steered --scenario-set no_country
```
