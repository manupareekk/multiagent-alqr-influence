# Multi-agent influence + A-LQR — results

**Student:** Manu Pareek  
**Advisor context:** Glen Chou / Trustworthy Robotics — apply lab A-LQR to calibrated multi-agent influence  
**Updated:** 2026-09-17  

---

## Goal

Agent **B** (LLM) should:

- **Resist** — when peer **A** pushes a *wrong* answer, keep the correct answer  
- **Accept** — when peer **A** pushes the *right* answer, take the correction  

Steering = lab **A-LQR** on B’s activations at generation time (after seeing A). Alone answers are unsteered. No fine-tuning.

**Lab A-LQR** (paper/repo) tested *single-agent* behaviors (toxicity, TruthfulQA truthfulness, refusal, concepts).  
**This project** tests *two-agent peer influence* with new influence pickles — same controller, different task.

---

## What I built / ran

1. Multi-agent harness (`multiagent/`) — scenarios, influence score, resist/accept split  
2. Influence pickles (contrastive + Jacobian) — first Gemma-2-2B, then **gemma-2-2b-it** matched to TQA-MC  
3. Unsteered baselines + A-LQR λ sweeps on Kaggle (T4/P100)  
4. Moved from handmade gold → TruthfulQA generation → **TruthfulQA MC1** (exact letter score)

**Live Agent A:** not run (gate was: steered resist must beat baseline first).

---

## Main result (headline)

On TruthfulQA MC + instruct model + matched pickles, **B copies bad peer A** (resist **1/10**).  
**A-LQR never improves resist** across λ = 0.5 … 12. Mild λ ≈ no change on resist; high λ **breaks** generation (garbage / repetition).  
Same negative pattern as earlier handmade / no-country runs.

---

## Experiment timeline

| # | What | Model / pickles | Outcome |
|---|------|-----------------|---------|
| 1 | Handmade (+ countries), λ=1.25 only | gemma-2-2b + early influence pickles | Steered **worse** (esp. resist) |
| 2 | `no_country` λ sweep | same | Every λ ≤ baseline; higher λ → worse resist |
| 3 | TQA open generation | gemma-2-2b-it | Better text after decode fix; gold match too soft → dropped |
| 4 | TQA MC1 baseline | gemma-2-2b-it, no steer | Resist **1/10**, accept **9/10** |
| 5 | Rebuild pickles | IT + TQA-MC prompts | Done (`influence-pickles-gemma2bit-tqa-mc.zip`) |
| 6 | λ sweep 0.5–2 | those pickles | Resist never beats baseline |
| 7 | High-λ check 3–12 | those pickles | Resist still fails; λ≥8 collapse |

---

## Table A — Handmade pilot (λ=1.25, n=20)

Artifact: `steered_compare_n20_20260910T025044Z.json`

| Slice | Baseline | Steered | Δ |
|-------|----------|---------|---|
| Overall | 14/20 | 11/20 | −3 |
| Resist bad peer | 6/11 | 3/11 | −3 |
| Accept good peer | 8/9 | 8/9 | 0 |

---

## Table B — `no_country` λ sweep (n=18)

Artifact: `lambda_sweep_no_country_20260917T020638Z.json`  
Model: gemma-2-2b + older influence pickles. Scoring: substring gold.

| λ | all | resist | accept |
|---|-----|--------|--------|
| baseline | 12/18 | 5/10 | 7/8 |
| 0.5 | 11/18 | 4/10 | 7/8 |
| 1.0 | 10/18 | 4/10 | 6/8 |
| 1.25 | 10/18 | 3/10 | 7/8 |
| 1.5 | 8/18 | 2/10 | 6/8 |
| 2.0 | 5/18 | 0/10 | 5/8 |

**Read:** Dropping country items didn’t fix it. Stronger λ hurt more.

---

## Table C — TQA-MC + gemma-2-2b-it (main)

**Setup:** 10 resist + 10 accept (MC1 letters). Scripted A. Score = correct letter after A.  
**Pickles:** `gemma-2-2b-it-{resist,defer,accept,ignore,influence_jac}.pkl`

Artifacts (Downloads):
- `lambda_sweep_tqa_mc_20260917T100715Z.json` — λ ∈ {0.5, 1.0, 1.25, 1.5, 2.0}
- `lambda_sweep_tqa_mc_20260917T103237Z.json` — λ ∈ {3, 5, 8, 12}

### C1 — Mild / mid λ

| λ | all | resist | accept |
|---|-----|--------|--------|
| baseline | 10/20 | 1/10 | 9/10 |
| 0.5 | 11/20 | 1/10 | 10/10 |
| 1.0 | 11/20 | 1/10 | 10/10 |
| 1.25 | 11/20 | 1/10 | 10/10 |
| 1.5 | 11/20 | 1/10 | 10/10 |
| 2.0 | 10/20 | 0/10 | 10/10 |

### C2 — High λ

| λ | all | resist | accept |
|---|-----|--------|--------|
| baseline | 10/20 | 1/10 | 9/10 |
| 3.0 | 10/20 | 0/10 | 10/10 |
| 5.0 | 10/20 | 0/10 | 10/10 |
| 8.0 | 0/20 | 0/10 | 0/10 |
| 12.0 | 0/20 | 0/10 | 0/10 |

### What the JSON shows (behavior)

- **Baseline resist:** alone often correct (~6/10); after bad A, flips to A’s letter (~5 flips). Classic overinfluence.  
- **λ 0.5–1.5:** almost same letters as baseline on resist; +1 accept (`accept_004` fixed).  
- **λ ≥ 3:** first letter often still wrong, then letter spam (`C C C…`).  
- **λ ≥ 8:** garbage (`wrong wrong…`, `Satoshi…`, `AssemblyCulture…`) — generation collapse, not “stronger resist.”

**Why garbage:** A-LQR sets target strength `β = λ × ‖E‖` per layer. Lab λ is ~1–2.5. Large λ warps activations; greedy decode + 80 new tokens → repetition loops.

---

## Combined read

| Finding | Evidence |
|---------|----------|
| Overinfluence is real | Alone right → after bad A wrong (MC) |
| Accept is easy | Baseline accept 9/10; mild steer +1 |
| Mild A-LQR doesn’t fix resist | Resist stuck at 1/10 for λ ≤ 1.5 |
| Stronger λ doesn’t help | Resist → 0; then total collapse |
| Not just “bad countries / one λ” | Handmade, no_country, and TQA-MC+IT all negative |
| λ ruled out as the fix | Swept 0.5 → 12 |

---

## vs lab A-LQR

| | Lab A-LQR | This project |
|--|-----------|--------------|
| Setting | Single model, one prompt | Two agents; B sees A’s message |
| Goals | Tox ↓, truth ↑, refusal, concepts | Resist bad peer / accept good peer |
| Pickles | tox / tqa / refusal / concepts | resist–defer, accept–ignore (+ jac) |
| Typical outcome in papers | Steering helps those tasks | On this peer-MC task, resist not helped |

Same math; **different job**. Negative result is about calibrated peer influence, not “A-LQR never works.”

---

## Artifacts checklist

| File | What |
|------|------|
| `~/Downloads/influence-pickles-gemma2bit-tqa-mc.zip` | IT + TQA-MC pickles |
| `~/Downloads/lambda_sweep_tqa_mc_20260917T100715Z.json` | MC λ 0.5–2 |
| `~/Downloads/lambda_sweep_tqa_mc_20260917T103237Z.json` | MC λ 3–12 |
| `~/Downloads/lambda_sweep_no_country_20260917T020638Z.json` | no_country sweep |
| `~/Downloads/steered_compare_n20_20260910T025044Z.json` | Handmade λ=1.25 |
| Kaggle code dataset | `lqr-m5` |

Code live in repo: `multiagent/`. Short status: `STATUS.md`.

---

## Next (agreed direction)

1. **Show Glen** this doc — ask soft A vs new peer-in-context pickles vs open-gen sanity.  
2. **Do not** more λ sweeps or live Agent A until resist beats baseline.  
3. Likely next experiment: **softer Agent A** messages; baseline + mild λ only (≤1.5).

---

## One-liner for meetings

> On TQA-MC with gemma-2-2b-it and matched influence pickles, Agent B copies bad peers (resist 1/10). A-LQR across λ 0.5–12 never raises resist; high λ only collapses generation. Need a method change, not another λ knob.
