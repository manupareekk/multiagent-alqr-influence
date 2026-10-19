# Multi-agent calibrated influence with A-LQR

Applying lab [A-LQR activation steering](https://github.com/trustworthyrobotics/lqr-activation-steering) to a **two-agent** setting: can closed-loop steering make Agent B *resist* bad peer pressure and *accept* good corrections?

**Finding.** On TruthfulQA MC1 with `gemma-2-2b-it` and matched influence pickles, B copies a hard wrong peer (resist **1/10**). Mild λ barely changes resist; large λ collapses generation. Handmade and `no_country` pilots showed the same pattern.

Builds on [Skifstad et al., A-LQR](https://arxiv.org/abs/2604.19018) (single-agent tox / truthfulness / refusal / concepts). This repo is the multi-agent influence harness, contrastive pickle builder, and eval results.

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

Contrastive sets use the same question + peer message with different B instructions (e.g. “only use A if correct” vs “always agree with A”). λ scales setpoint strength.

---

## Results

### TruthfulQA MC1 + gemma-2-2b-it (main)

Artifacts: `results/lambda_sweep_tqa_mc_20260917T100715Z.json`, `...T103237Z.json`

| λ | all | resist | accept |
|---|-----|--------|--------|
| baseline | 10/20 | 1/10 | 9/10 |
| 0.5–1.5 | 11/20 | 1/10 | 10/10 |
| 2–5 | 10/20 | 0/10 | 10/10 |
| 8–12 | 0/20 | 0/10 | 0/10 |

Alone, B often knows the right letter; after a confident wrong peer it flips. Mild steering almost never fixes resist. High λ produces junk / repetition, not better independence.

### Earlier pilots

**Handmade** (`results/steered_compare_n20_20260910T025044Z.json`) — gemma-2-2b, λ=1.25: 14/20 → 11/20; resist 6/11 → 3/11.

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

- Pickle prompts were originally plain text while IT eval used chat templates; `--chat-template` was added later for matched rebuilds.
- Live (non-scripted) Agent A was not evaluated — resist never beat baseline under scripted hard A.
- Higher λ is not a fix here; it degrades fluency before helping resist.

Possible follow-ons: chat-template-matched pickles + short MC decode; peer-in-context methods beyond A-LQR (e.g. runtime gates or SFT) on the same resist/accept harness.

---

## Citation

Upstream A-LQR: [arXiv:2604.19018](https://arxiv.org/abs/2604.19018) · [lab code](https://github.com/trustworthyrobotics/lqr-activation-steering)
