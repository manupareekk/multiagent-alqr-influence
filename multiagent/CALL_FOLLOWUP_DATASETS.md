# Follow-up for Glen — contrastive datasets / v

Quick clarification from the call — I think I answered the wrong question.

When you asked about the two datasets, I thought you meant my eval split (resist vs accept on TruthfulQA-MC). What you meant was the contrastive pair used to build the steering direction — i.e. the feature / v vector. Sorry about the mix-up.

What I’m doing: the steering objective is calibrated influence on Agent B. Resist means don’t copy a bad peer A; accept means take a good peer A’s correction. For resist, the two datasets are the same question and bad A message, but with different instructions to B — one says only use A if correct, the other says always agree with A (defer). Accept vs ignore is the same idea with a good A. From that I get E = acts(desired) − acts(contrast), v = E normalized, and λ controls how hard we push. The eval is separate: hard scripted A, and I score whether B outputs the right letter after seeing A.

Right now B copies bad A a lot (resist is about 1/10). Mild λ barely helps on resist, and large λ just breaks the generation. So the pipeline runs, but this feature isn’t fixing hard peer pressure.

Next I’m going to (1) rebuild the contrastive prompts with the same Gemma chat template I use at eval and rerun mild λ, (2) shorten max tokens on MC so we aren’t scoring junk completions, and (3) check whether this instruction-defined contrast actually moves the letter or logprobs. If that still fails on hard A, I’ll keep the multi-agent goal and talk next method with you — not more λ sweeps.

Does that match what you were asking on the call?
