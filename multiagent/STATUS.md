# Multi-agent influence + A-LQR — status

**Date:** 2026-09-17  
**Student:** Manu Pareek  
**Goal:** Calibrated influence — Agent B resists bad peer A, accepts good A. Steer B with lab A-LQR.

---

## Timeline (what we ran)

1. **Handmade set (with country items)** — Gemma-2-2B base, old pickles, λ=1.25: steered worse (resist 6→3).

2. **`no_country` λ sweep** — every λ ≤ baseline; higher λ → worse resist.

3. **TQA generation + gemma-2-2b-it** — better gens; gold match noisy → dropped for MC.

4. **TQA MC1 + IT baseline** — resist **1/10**, accept **9/10**. Alone often right; after bad A, flips.

5. **Pickles IT + TQA-MC** — Done (`influence-pickles-gemma2bit-tqa-mc.zip`).

6. **λ sweep on those pickles** — **Done** (see below). Resist never improved. Live A **not** next.

---

## Where we are

| Step | Status |
|------|--------|
| Handmade + countries (λ=1.25) | Done — steered worse |
| `no_country` λ sweep | Done — still negative |
| TQA generation + IT | Done — dropped for MC |
| TQA MC1 + IT baseline | Done — resist 1/10, accept 9/10 |
| Pickles (IT + TQA-MC) | Done |
| λ sweep (IT + TQA-MC) 0.5–2 | **Done** — resist flat / worse at λ=2 |
| High-λ check 3–12 | **Done** — resist never improves; λ≥8 collapses |
| Live Agent A | **Hold** |

---

## TQA-MC λ sweeps (2026-09-17)

Artifacts:
- `lambda_sweep_tqa_mc_20260917T100715Z.json` (0.5–2)
- `lambda_sweep_tqa_mc_20260917T103237Z.json` (3–12)

| λ | all | resist | accept |
|---|-----|--------|--------|
| baseline | 10/20 | 1/10 | 9/10 |
| 0.5–1.5 | 11/20 | 1/10 | 10/10 |
| 2–5 | 10/20 | 0/10 | 10/10 |
| 8–12 | 0/20 | 0/10 | 0/10 |

**Read:** Mild λ ≈ no change on resist (+1 accept). Higher λ hurts resist, then breaks all answers. λ is ruled out as the fix.

---

## What we’re doing now

Glen OK’d integrity check. **Next run:** rebuild IT pickles with **chat template**, then mild λ sweep on TQA-MC with **short max tokens** (`--max-new-tokens 8`). Code zip: `~/Downloads/lqr-ma-chatfix.zip`.

---

## Blockers / watchouts

- pip `torchvision` messages = ignore (Kaggle junk).  
- Don’t reuse old `gemma-2-2b` pickles on IT.  
- Live A only if resist beats baseline without killing accept — **not met**.

---

## Full write-up

See **`RESULTS.md`** — timeline, all tables (handmade / no_country / TQA-MC λ 0.5–12), artifacts, next steps.

## One-liner

λ 0.5→12 on IT+TQA-MC: resist never beats 1/10; mild λ ≈ noop on resist; λ≥8 destroys answers. Stop sweeping λ — need a method change with Glen.
