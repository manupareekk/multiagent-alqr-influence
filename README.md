# Multi-agent calibrated influence with A-LQR

Applying lab [A-LQR activation steering](https://github.com/trustworthyrobotics/lqr-activation-steering) to a **two-agent** setting: can closed-loop steering make Agent B *resist* bad peer pressure and *accept* good corrections?

**Finding.** Peer compliance is not a linearly steerable direction. On TruthfulQA MC1 with `gemma-2-2b-it` and matched influence pickles, B answers most resist items correctly *alone* yet retains the correct answer after an assertive wrong peer in only **1/10** cases — acute overinfluence, not ignorance. A-LQR steering never improves resist across λ ∈ [0.5, 12]: mild λ leaves it unchanged, moderate λ eliminates it, high λ collapses decoding. On a pilot sweep, steering on several items moved B *toward* the incorrect peer. Handmade and `no_country` pilots showed the same pattern.

Builds on [Skifstad et al., A-LQR](https://arxiv.org/abs/2604.19018) (single-agent tox / truthfulness / refusal / concepts). This repo is the multi-agent influence harness, contrastive pickle builder, and eval results. Accompanying writeup: `multiagent_alqr_paper.tex` (technical report).

---

## Setup

| Role | Role in eval |
|------|----------------|
| **Agent A** | Scripted peer (wrong letter on resist; correct on accept) |
| **Agent B** | HF model; answers alone, then after seeing A |

| Objective | Success |
|-----------|---------|
| **Resist** | After bad A, B still outputs the correct MC letter |
| **Accept** | After good A, B outputs the correct MC letter |

Steering = A-LQR on B **after** A only. Directions from contrastive activations:

- \(X_\text{resist} = \mathrm{acts}(\text{resist}) - \mathrm{acts}(\text{defer})\)
- \(X_\text{accept} = \mathrm{acts}(\text{accept}) - \mathrm{acts}(\text{ignore})\)

Contrastive sets use the same question + peer message with different B instructions (e.g. “only use A if correct” vs “always agree with A”). λ scales setpoint strength (β = λ × ‖E‖ per layer).

The harness also records a continuous influence score per item: the shift in B's output toward A after peer exposure, `sim(B_after, A) − sim(B_alone, A)`.

---

## Results

### TruthfulQA MC1 + gemma-2-2b-it (main)

Artifacts: `results/lambda_sweep_tqa_mc_20260917T100715Z.json`, `...T103237Z.json`

| Condition | all | resist | accept | Generation quality |
|-----------|-----|--------|--------|--------------------|
| Alone (no peer, unsteered) | — | ~6/10 | — | Pre-peer capability mostly intact |
| Peer, unsteered (baseline) | 10/20 | 1/10 | 9/10 | Fluent; acute overinfluence |
| A-LQR λ 0.5–1.5 | 11/20 | 1/10 | 10/10 | Fluent; no resist gain |
| A-LQR λ 2–5 | 10/20 | 0/10 | 10/10 | Letter spam; zero resist |
| A-LQR λ 8–12 | 0/20 | 0/10 | 0/10 | Severe decoding collapse |

**Read:** the alone-vs-peer gap (~6/10 → 1/10 on resist) shows this is a social-compliance failure, not a knowledge gap. Accept is already near-ceiling at baseline, so all measurable harm concentrates on resist — where steering is neutral-to-negative at every λ. High λ does not produce stronger resistance; it warps hidden states off the generation manifold (letter spam, then collapse) before affecting social behavior.

### Earlier pilots

**Handmade** (`results/steered_compare_n20_20260910T025044Z.json`) — gemma-2-2b, λ=1.25: 14/20 → 11/20; resist 6/11 → 3/11; accept 8/9 → 8/9. On several resist items, steering *increased* movement toward the incorrect peer — the push landed on the wrong side of the compliance direction.

**no_country λ sweep** (`results/lambda_sweep_no_country_20260917T020638Z.json`):

| λ | all | resist | accept |
|---|-----|--------|--------|
| baseline | 12/18 | 5/10 | 7/8 |
| 0.5 | 11/18 | 4/10 | 7/8 |
| 1.0 | 10/18 | 4/10 | 6/8 |
| 1.25 | 10/18 | 3/10 | 7/8 |
| 1.5 | 8/18 | 2/10 | 6/8 |
| 2.0 | 5/18 | 0/10 | 5/8 |

---

## Interpretation

- Contrastive vectors cleanly shift *single-agent* traits, but peer-induced deference appears entangled with instruction-following and the token distribution itself — steering toward "resist" did not selectively reduce overinfluence.
- The λ knob is ruled out across 0.5–12. This is a **method limitation, not a tuning gap** — calibrated peer influence likely requires non-linear or gated interventions (e.g., steering conditioned on detected sycophancy) or multi-agent SFT.
- Negative result scoped to this regime: 2B model, scripted hard peer, exact-letter MC. It does not claim A-LQR "never works" — same math, different job.

---

## Repo

```
multiagent/   # scenarios, metrics, backends, influence_data, λ sweep
results/      # JSON run dumps
patches/      # small steer/ diffs (fp16, decode-only new tokens)
```

Depends on the lab `steer/` package (clone [lqr-activation-steering](https://github.com/trustworthyrobotics/lqr-activation-steering) and place `multiagent/` alongside, or merge). Apply `patches/steer-fp16-decode-signal.patch` if you hit dtype / prompt-echo issues on Gemma.

Influence pickles (hundreds of MB) are not in git — rebuild on GPU:

```bash
python -m multiagent.influence_data \
  --model gemma2bit --prompt-source tqa \
  --pickle-dir ./pickles_it_tqa_mc --n-jac 30 --chat-template
```

Eval / λ sweep:

```bash
python -m multiagent.run_lambda_sweep \
  --baseline alqr --steered alqr \
  --hf-model google/gemma-2-2b-it --model-key gemma-2-2b-it \
  --pickle-dir ./pickles_it_tqa_mc --no-quant \
  --scenario-set tqa_mc \
  --tqa-n-resist 10 --tqa-n-accept 10 --tqa-seed 0 \
  --lambdas 0.5,1.0,1.25,1.5 \
  --max-new-tokens 8
```

More notes: `multiagent/RESULTS.md`, `multiagent/KAGGLE.md`.

---

## Limitations / open threads

- n=20 per condition, one 2B model, one seed — consistent replication across three dataset/model variants mitigates but does not eliminate this.
- Live (non-scripted) Agent A was not evaluated — deliberately, until steered resist beats baseline.
- Prompt-engineering baseline (e.g. "ignore the peer if unsure") not yet run; would isolate whether the failure is specific to activation-level methods.
- Pickle prompts were originally plain text while IT eval used chat templates; `--chat-template` matched rebuilds stabilized extraction without changing the outcome.
- Higher λ is not a fix here; it degrades fluency before helping resist.

Possible follow-ons: prompt baseline + softer peer A on the same harness; gated/nonlinear steering; larger n on TruthfulQA MC1 (817 items available).

---

## Citation

Upstream A-LQR: [arXiv:2604.19018](https://arxiv.org/abs/2604.19018) · [lab code](https://github.com/trustworthyrobotics/lqr-activation-steering)

Work carried out as part of a research project with the Trustworthy Robotics Lab, Georgia Institute of Technology.
