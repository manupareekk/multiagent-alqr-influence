# Next steps: TruthfulQA + gemma-2-2b-it

Do **not** jump to live agents yet. Order:

1. **Baseline smoke** (TQA + instruct model, no steering) — check answers look sane  
2. **Rebuild influence pickles** for `gemma-2-2b-it` on TQA prompts  
3. **λ sweep** on TQA scenarios  
4. Live Agent A only if (3) beats baseline

---

## On your laptop

```bash
cd ~/lqr-activation-steering
# zip already refreshed as ~/Downloads/lqr-ma.zip after code update
```

Re-upload **`lqr-ma.zip`** into Kaggle dataset `lqr-maaa` (replace old files).

Accept HF license: https://huggingface.co/google/gemma-2-2b-it

---

## Kaggle Cell 1 — TQA baseline (no pickles, ~15–30 min)

Factory reset if GPU dirty. Attach `lqr-maaa`. Internet on. GPU on.

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
    if p.is_dir() and p.name.lower() in ("lqr-maaa", "lqr-maa", "lqr-ma"):
        src = p
        break
assert src, "attach lqr-maaa"

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
assert Path("multiagent/run_tqa_baseline.py").exists()

from huggingface_hub import login
login(token="HF_TOKEN_HERE")  # paste token

!python -m multiagent.run_tqa_baseline \
  --hf-model google/gemma-2-2b-it \
  --n-resist 10 --n-accept 10 --seed 0
```

**Pass gate:** read 3 sample B answers. If they look like normal short answers (not Agent A/B loops), continue. Paste SUCCESS line + 1 sample here.

---

## Kaggle Cell 2 — rebuild pickles (slow, once; ~few hours)

Same session OK if VRAM free; else factory reset, re-copy code, then:

```python
%cd /kaggle/working/lqr-activation-steering
!mkdir -p /kaggle/working/pickles_it_tqa
!python -m multiagent.influence_data \
  --model gemma2bit \
  --prompt-source tqa \
  --pickle-dir /kaggle/working/pickles_it_tqa \
  --n-jac 30
!cd /kaggle/working && zip -r influence-pickles-gemma2bit-tqa.zip pickles_it_tqa
```

Download that zip before the session dies.

---

## Kaggle Cell 3 — λ sweep on TQA + it model

```python
%cd /kaggle/working/lqr-activation-steering
!python -m multiagent.run_lambda_sweep \
  --baseline alqr --steered alqr \
  --hf-model google/gemma-2-2b-it --model-key gemma-2-2b-it \
  --pickle-dir /kaggle/working/pickles_it_tqa --no-quant \
  --scenario-set tqa \
  --tqa-n-resist 10 --tqa-n-accept 10 --tqa-seed 0 \
  --lambdas 0.5,1.0,1.25,1.5,2.0
```

---

## What changed in code

- `multiagent/tqa_scenarios.py` — TruthfulQA → resist/accept scenarios  
- `run_tqa_baseline.py` — instruct model + chat template, no steering  
- `influence_data --model gemma2bit --prompt-source tqa`  
- Chat template in ALQR / HF backends  
- Scoring supports multiple TQA `correct_answers`
