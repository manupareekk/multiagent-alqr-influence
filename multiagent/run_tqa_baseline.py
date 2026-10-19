#!/usr/bin/env python3
"""TruthfulQA MC1 + instruct model baseline (NO steering / NO pickles).

Exact letter scoring. Verifies generations before rebuilding pickles.

Kaggle:
  python -m multiagent.run_tqa_baseline \\
      --hf-model google/gemma-2-2b-it \\
      --scenario-set tqa_mc \\
      --n-resist 10 --n-accept 10 --seed 0
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .run_steered import eval_backend
from .scenarios import get_scenarios
from .steered_backend import make_steerable_backend


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--hf-model", default="google/gemma-2-2b-it")
    p.add_argument("--n-resist", type=int, default=10)
    p.add_argument("--n-accept", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--scenario-set",
        default="tqa_mc",
        choices=["tqa_mc", "tqa", "tqa_gen"],
        help="tqa_mc/tqa = MC1 exact letter; tqa_gen = legacy generation",
    )
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "runs"),
    )
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    scenarios = get_scenarios(
        scenario_set=args.scenario_set,
        tqa_n_resist=args.n_resist,
        tqa_n_accept=args.n_accept,
        tqa_seed=args.seed,
    )
    print(
        f"TQA baseline  set={args.scenario_set}  model={args.hf_model}  "
        f"n={len(scenarios)}  resist={args.n_resist} accept={args.n_accept}"
    )
    print("Example item:\n", scenarios[0].user_question[:400], "\n---")

    tok = AutoTokenizer.from_pretrained(args.hf_model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="eager",
    )
    model.eval()

    backend = make_steerable_backend(
        "hf_chat",
        model_obj=model,
        tokenizer=tok,
        use_chat_template=True,
    )
    result = eval_backend("tqa_baseline", backend, scenarios)

    resist = [
        r
        for r in result["rows"]
        if r["scenario"]["objective"] == "resist_overinfluence"
    ]
    accept = [
        r
        for r in result["rows"]
        if r["scenario"]["objective"] == "accept_good_influence"
    ]
    print("=" * 72)
    print(
        f"SUCCESS  all {result['n_success']}/{result['n']}  "
        f"resist {sum(1 for r in resist if r['metrics']['success'])}/{len(resist)}  "
        f"accept {sum(1 for r in accept if r['metrics']['success'])}/{len(accept)}"
    )
    print("Sample generations:")
    for r in result["rows"][:3]:
        gold = r["scenario"].get("mc_gold_letter") or r["scenario"]["gold_answer"]
        print(f"\n[{r['scenario']['id']}] gold={gold}")
        print(f"  A: {r['agent_a'][:120]}")
        print(f"  B alone: {r['b_alone'][:120]}")
        print(f"  B after: {r['b_after'][:120]}")
        print(f"  metrics: {r['metrics']}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"tqa_mc_baseline_{stamp}.json"
    payload = {
        "created_utc": stamp,
        "hf_model": args.hf_model,
        "scenario_set": args.scenario_set,
        "n_resist": args.n_resist,
        "n_accept": args.n_accept,
        "seed": args.seed,
        "result": result,
        "scenario_preview": [asdict(s) for s in scenarios[:2]],
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
