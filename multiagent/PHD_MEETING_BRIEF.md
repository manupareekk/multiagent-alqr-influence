# PhD Update: Multi-Agent Influence + A-LQR

**Student:** Manu Pareek  
**Context:** First milestone under Glen Chou / Trustworthy Robotics — A-LQR (Skifstad) applied to multi-agent influence  
**Date:** Sep 10, 2026  
**Artifacts:** `influence-pickles.zip`, `steered_compare_n20_20260910T025044Z.json`, `multiagent/` harness

---

## 1. What to say (meeting script)

### Open (45–60 sec)
I’ve been getting the lab A-LQR stack running for a multi-agent follow-on. The goal isn’t “make the model less toxic” — it’s **calibrated influence**: Agent B should **resist bad peer pressure** and **accept good corrections**. I built a small eval harness, generated Gemma-2-2B **influence pickles** on Kaggle, and ran steered vs unsteered compares.

### What I built (1–2 min)
Three pieces:

1. **Harness** — scenarios with gold answers, plus a continuous influence score: how much B moves toward A after seeing A.
2. **Influence pickles** — resist/defer/accept/ignore activations + Jacobians for Gemma-2-2B (not the toxicity pickles).
3. **A-LQR on Agent B** at generation time, compared to a **baseline** with no steering.

Important: we’re **not fine-tuning** Gemma. Offline we build steering directions; online we run closed-loop control.

### Results (be blunt, 1 min)
On **n=20**, λ=1.25:

- **Baseline: 14/20**
- **Steered: 11/20** (Δ −3)

That sounds like “steering failed,” but the split matters:

- **Accept good A:** 8/9 → 8/9 (unchanged)
- **Resist bad A:** 6/11 → 3/11 (all the damage)

On some resist cases, influence **toward the bad peer went up** after steering — opposite of what we want. So the pipeline works; this controller setting is miscalibrated for resist.

### Why that’s still a good milestone (30 sec)
I now have: a measurable multi-agent task, real pickles, and a first scientific signal. The next work is **tuning**, not rebuilding from scratch.

### Ask / close (30 sec)
Does this framing match what you wanted? Should I prioritize a **λ sweep** on these pickles (cheap), or rebuild richer influence pickles first (expensive)?

---

## 2. Numbers cheat sheet

| Slice | Baseline | Steered | Δ |
|-------|----------|---------|---|
| Overall | 14/20 | 11/20 | −3 |
| Resist bad peer (11) | 6/11 | 3/11 | −3 |
| Accept good critique (9) | 8/9 | 8/9 | 0 |

**Broke (resist only):** `false_capital`, `false_planet`, `false_chromosomes`  
**Accept:** no scenarios fixed, none broken  

**Mean influence:** 0.103 → 0.067  

**Files to have ready:**
- `~/Downloads/steered_compare_n20_20260910T025044Z.json`
- `~/Downloads/influence-pickles.zip`
- Local `multiagent/` harness + updated `lqr-ma.zip`

---

## 3. Why this matters / significance

1. **Operates the lab stack** on free GPU (Gemma-2-2B, pickles, Jacobians, steering).
2. **Defines a measurable multi-agent problem** (resist vs accept) with a continuous influence signal.
3. **Separates “method runs” from “method helps”** — negative pilot is real science.
4. **Lab narrative:** prior A-LQR steers one model’s behavior (tox/refusal/etc.); this attaches **closed-loop control to Agent B in a two-agent interaction** for calibrated influence.

Not inventing multi-agent from scratch. Novel piece is **A-LQR on Agent B for peer influence**, with a measurable harness.

---

## 4. Why steering might not have helped

A-LQR does not retrain weights. It nudges activations toward an offline direction. That can fail if:

1. **Wrong strength (λ)** — too strong or too weak (we only tried 1.25).
2. **Noisy / misaligned resist vector** — small prompt set for pickles.
3. **Wrong-way push** — on several resist cases, influence toward bad A *increased* after steering.
4. **Strict success metric** — e.g. gold kept but influence above threshold → counted as fail.
5. **Accept already easy** — baseline 8/9; little upside; harm concentrated on resist.

---

## 5. What is the “cheap λ sweep”?

**λ (lambda)** = how hard A-LQR pushes toward the setpoint.

| | Cheap | Expensive |
|--|--------|-----------|
| What | Same pickles, re-run eval at λ = 0.5, 1.0, 1.25, 2.0, … | Rebuild pickles / more Jacobians |
| Cost | Hours of GPU | Many more GPU hours |
| Goal | Find λ where resist improves without hurting accept | Better steering directions |

Always report **resist vs accept separately**.

---

## 6. What to do next (priority)

1. λ sweep on fixed pickles + n=20; plot success vs λ for resist and accept.
2. Report “keeps gold” separately from “low influence” on resist failures.
3. Add a prompt-only baseline (“ignore Agent A if unsure”) for comparison.
4. Rebuild richer influence pickles only after a λ looks promising.
5. Longer-term: PACE GPU, stronger judge, larger scenario set.

---

## 7. Q&A — expected questions and answers

**Q: Did you train the model?**  
A: No. A-LQR doesn’t update weights. I built contrastive activation pickles + Jacobians offline, then applied closed-loop steering at inference.

**Q: What’s baseline?**  
A: Same model, same prompts, same Agent A messages — but **no** activation steering. Vanilla generation.

**Q: Why didn’t steering help?**  
A: Most likely λ / direction miscalibration. Accept was already strong (8/9). Resist got worse, and on some items B moved *more* toward bad A. Pipeline works; controller settings for resist are wrong.

**Q: Isn’t a negative result bad?**  
A: It’s informative. The eval detects harm, and overall accuracy hides resist vs accept. That’s what we want before claiming gains.

**Q: How is influence measured?**  
A: Continuously: similarity(B_after, A) − similarity(B_alone, A). Success also requires gold (or refuse) after seeing A, with a cap on over-influence for resist cases.

**Q: Why Gemma-2-2B / Kaggle?**  
A: No PACE yet. Free Kaggle GPU is enough for a 2B fp16 pilot. Happy to move to lab/PACE when available.

**Q: How is this different from lab tox work?**  
A: Tox pickles steer toxic vs nontoxic. These are **influence** pickles: resist/defer/accept/ignore under peer messages. Same A-LQR machinery, different objective and eval.

**Q: Why only 20 scenarios?**  
A: First pilot for signal + pipeline. Enough to see resist vs accept split. Next: λ sweep on 20, then grow the set if a setting looks promising.

**Q: What’s the cheap next experiment?**  
A: λ sweep — same pickles, same n=20, try λ like 0.5 / 1.0 / 1.25 / 2.0.

**Q: What’s expensive next?**  
A: Rebuild pickles with more/better influence prompts, or bigger models / more jac samples.

**Q: Could your metric be wrong?**  
A: Partly possible — e.g. one case still had gold but failed the influence threshold. I’ll sanity-check failures qualitatively and report “keeps gold” separately from “low influence.”

**Q: What do you want from me?**  
A: Confirm resist/accept framing; preference for λ sweep vs rebuild pickles first; help with PACE if useful; whether to add a prompt-only baseline.

**Q: How long did this take?**  
A: About a week of onboarding + harness, then a heavy Kaggle stretch for influence pickles and n=20 eval. A few GPU hours of compute; a lot of environment debugging.

**Q: What’s novel?**  
A: Not inventing multi-agent from scratch. Novel piece is A-LQR closed-loop control on Agent B for calibrated peer influence, with a measurable harness.

**Q: What if λ sweep also fails?**  
A: Then the resist vector is likely misaligned — rebuild influence data / revisit resist vs defer, and compare to prompt-only baselines.

---

## 8. If he pushes / sounds skeptical

> Totally fair that 14→11 isn’t a win. I’m treating it as a calibration failure on resist, not as “A-LQR can’t work here.” The split makes that concrete: accept stable, resist degraded. I’ll come back with a λ curve and resist/accept broken out.

---

## 9. Do / don’t

**Do say**
- Pipeline works
- Negative pilot; resist took the hit
- Next = λ sweep + split metrics

**Don’t say**
- “I trained Gemma”
- “Steering works”
- Hide the −3
- Overclaim novelty

---

## 10. One-line summary

**Capability unlocked (harness + influence pickles + A-LQR on B); first n=20 signal is negative overall because resist degraded while accept held; next step is cheap λ sweeps with resist/accept reported separately.**
