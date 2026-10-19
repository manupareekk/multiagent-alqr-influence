#!/usr/bin/env python3
"""λ sweep on fixed influence pickles (scripted A).

Diagnoses whether steering strength is too high/low vs dataset noise.
Default eval set is ``no_country`` (drops Australia/Canada capital items).

Local smoke:
  python -m multiagent.run_lambda_sweep --baseline dry_run --steered dry_run_steered

Kaggle (reuse one Gemma load):
  python -m multiagent.run_lambda_sweep \\
      --baseline alqr --steered alqr \\
      --hf-model google/gemma-2-2b --model-key gemma-2-2b \\
      --pickle-dir /kaggle/working/pickles --no-quant \\
      --scenario-set no_country \\
      --lambdas 0.5,1.0,1.25,1.5,2.0
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from .run_steered import eval_backend
from .scenarios import get_scenarios
from .steered_backend import make_steerable_backend


def _split_stats(result: dict) -> dict:
    resist = [r for r in result["rows"] if r["scenario"]["objective"] == "resist_overinfluence"]
    accept = [r for r in result["rows"] if r["scenario"]["objective"] == "accept_good_influence"]

    def pack(rows: list) -> dict:
        n = len(rows)
        n_ok = sum(1 for r in rows if r["metrics"]["success"])
        mean_inf = (
            sum(r["metrics"]["influence"] for r in rows) / n if n else 0.0
        )
        return {"n": n, "n_success": n_ok, "mean_influence": mean_inf}

    return {"overall": pack(result["rows"]), "resist": pack(resist), "accept": pack(accept)}


def _print_split(label: str, split: dict) -> None:
    o, r, a = split["overall"], split["resist"], split["accept"]
    print(
        f"{label}: "
        f"all {o['n_success']}/{o['n']}  "
        f"resist {r['n_success']}/{r['n']}  "
        f"accept {a['n_success']}/{a['n']}  "
        f"mean_inf={o['mean_influence']:+.3f}"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="dry_run")
    p.add_argument("--steered", default="dry_run_steered")
    p.add_argument("--hf-model", default=None)
    p.add_argument("--model-key", default=None)
    p.add_argument("--pickle-dir", default=None)
    p.add_argument("--no-quant", action="store_true")
    p.add_argument(
        "--scenario-set",
        default="no_country",
        choices=["all", "no_country", "tqa", "tqa_mc", "tqa_gen"],
        help="tqa/tqa_mc = TruthfulQA MC1; tqa_gen = generation split",
    )
    p.add_argument("--tqa-n-resist", type=int, default=10)
    p.add_argument("--tqa-n-accept", type=int, default=10)
    p.add_argument("--tqa-seed", type=int, default=0)
    p.add_argument(
        "--lambdas",
        default="0.5,1.0,1.25,1.5,2.0",
        help="Comma-separated λ values for steered runs",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=80,
        help="Generation length (use ~4–8 for MC letter answers)",
    )
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "runs"),
    )
    args = p.parse_args()

    lambdas = [float(x.strip()) for x in args.lambdas.split(",") if x.strip()]
    scenarios = get_scenarios(
        scenario_set=args.scenario_set,
        tqa_n_resist=args.tqa_n_resist,
        tqa_n_accept=args.tqa_n_accept,
        tqa_seed=args.tqa_seed,
    )
    print(
        f"scenario_set={args.scenario_set}  n={len(scenarios)}  "
        f"lambdas={lambdas}  max_new_tokens={args.max_new_tokens}"
    )

    # Shared HF model for baseline + all λ (alqr only).
    shared_model = shared_tok = None
    if args.baseline == "alqr" or args.steered == "alqr":
        if not args.hf_model or not args.pickle_dir:
            raise SystemExit("alqr requires --hf-model and --pickle-dir")
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading shared HF model once…")
        shared_tok = AutoTokenizer.from_pretrained(args.hf_model)
        if shared_tok.pad_token is None:
            shared_tok.pad_token = shared_tok.eos_token
            shared_tok.pad_token_id = shared_tok.eos_token_id
        load_kw = {
            "torch_dtype": torch.float16,
            "device_map": "auto",
            "attn_implementation": "eager",
        }
        shared_model = AutoModelForCausalLM.from_pretrained(args.hf_model, **load_kw)
        shared_model.eval()
        print("shared model ready")

    print("=" * 72)
    # Prefer a single alqr backend (baseline = objective none, then λ sweeps).
    if args.baseline == "alqr" and args.steered == "alqr":
        backend = make_steerable_backend(
            "alqr",
            model=args.hf_model,
            pickle_dir=args.pickle_dir,
            model_key=args.model_key,
            quant=not args.no_quant,
            objective_mode="none",
            model_obj=shared_model,
            tokenizer=shared_tok,
            max_new_tokens=args.max_new_tokens,
        )
        print("BASELINE (unsteered)")
        backend.objective_mode = "none"
        baseline = eval_backend("baseline", backend, scenarios)
        base_split = _split_stats(baseline)
        _print_split("baseline", base_split)

        steered_backend = backend
        steered_backend.objective_mode = "auto"
    else:
        print("BASELINE (unsteered)")
        base_backend = make_steerable_backend(
            args.baseline,
            model=args.hf_model,
            pickle_dir=args.pickle_dir,
            model_key=args.model_key,
            quant=not args.no_quant,
            objective_mode="none",
            model_obj=shared_model,
            tokenizer=shared_tok,
            max_new_tokens=args.max_new_tokens,
        )
        baseline = eval_backend("baseline", base_backend, scenarios)
        base_split = _split_stats(baseline)
        _print_split("baseline", base_split)

        steered_backend = make_steerable_backend(
            args.steered,
            model=args.hf_model,
            pickle_dir=args.pickle_dir,
            model_key=args.model_key,
            quant=not args.no_quant,
            objective_mode="auto",
            lmbda_resist=lambdas[0],
            lmbda_accept=lambdas[0],
            model_obj=shared_model,
            tokenizer=shared_tok,
            max_new_tokens=args.max_new_tokens,
        )

    steered_runs = []
    for lam in lambdas:
        print("=" * 72)
        print(f"STEERED λ={lam}")
        if hasattr(steered_backend, "lmbda_resist"):
            steered_backend.lmbda_resist = lam
            steered_backend.lmbda_accept = lam
        if hasattr(steered_backend, "objective_mode"):
            steered_backend.objective_mode = "auto"
        steered = eval_backend(f"steered_l{lam}", steered_backend, scenarios)
        split = _split_stats(steered)
        _print_split(f"λ={lam}", split)
        steered_runs.append(
            {
                "lambda": lam,
                "summary": split,
                "n_success": steered["n_success"],
                "mean_influence": steered["mean_influence"],
                "rows": steered["rows"],
            }
        )

    print("=" * 72)
    print("SUMMARY (resist / accept split)")
    print(
        f"{'λ':>6}  {'all':>7}  {'resist':>8}  {'accept':>8}  {'Δall':>5}  {'Δresist':>8}"
    )
    b_all = base_split["overall"]["n_success"]
    b_res = base_split["resist"]["n_success"]
    for run in steered_runs:
        s = run["summary"]
        print(
            f"{run['lambda']:6.2f}  "
            f"{s['overall']['n_success']:2d}/{s['overall']['n']:<2d}   "
            f"{s['resist']['n_success']:2d}/{s['resist']['n']:<2d}    "
            f"{s['accept']['n_success']:2d}/{s['accept']['n']:<2d}    "
            f"{s['overall']['n_success'] - b_all:+3d}    "
            f"{s['resist']['n_success'] - b_res:+3d}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"lambda_sweep_{args.scenario_set}_{stamp}.json"
    payload = {
        "created_utc": stamp,
        "scenario_set": args.scenario_set,
        "scenario_ids": [s.id for s in scenarios],
        "lambdas": lambdas,
        "max_new_tokens": args.max_new_tokens,
        "baseline": {"summary": base_split, "rows": baseline["rows"]},
        "steered": steered_runs,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
