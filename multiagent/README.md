"""
Multi-agent influence harness + A-LQR steering on Agent B

Phase 1 — measure influence (no pickles)
----------------------------------------
  python3 -m multiagent.runner --backend dry_run
  python3 -m multiagent.runner --backend dry_run_calibrated

Phase 2 — steered vs baseline (local smoke, simulated steering)
---------------------------------------------------------------
  python3 -m multiagent.run_steered --baseline dry_run --steered dry_run_steered

Phase 3 — real A-LQR on Kaggle/PACE
-----------------------------------
  # build NEW influence pickles (not toxicity pickles)
  python -m multiagent.influence_data --model gemma2b --pickle-dir ./pickles --quant

  # compare unsteered vs steered Agent B
  python -m multiagent.run_steered \\
      --baseline alqr --steered alqr \\
      --baseline-objective none --steered-objective auto \\
      --hf-model google/gemma-2-2b --model-key gemma-2-2b \\
      --pickle-dir ./pickles

See KAGGLE.md for notebook cells.

Files
-----
  scenarios.py          eval scenarios (resist / accept)
  influence_prompts.py  expanded prompts for vector building
  influence_data.py     builds resist/defer/accept/ignore + jac pickles
  steered_backend.py    DryRunSteered + ALQRSteeredBackend
  run_steered.py        baseline vs steered comparison
  metrics.py            continuous influence score
  runner.py             measurement-only loop

Toxicity pickles (gemma-2-2b-tox etc.)
-------------------------------------
Keep for single-agent A-LQR repro. Do NOT use them here.
Influence steering needs resist/defer/accept/ignore pickles from influence_data.py.
"""
