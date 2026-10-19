#!/usr/bin/env python3
"""Compare unsteered vs steered Agent B on influence scenarios.

Local (no GPU):
  python -m multiagent.run_steered --baseline dry_run --steered dry_run_steered

Kaggle (after influence_data pickles):
  python -m multiagent.run_steered \\
      --baseline dry_run \\
      --steered alqr \\
      --hf-model google/gemma-2-2b \\
      --model-key gemma-2-2b \\
      --pickle-dir /kaggle/working/pickles
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .metrics import score_scenario
from .scenarios import get_scenarios
from .steered_backend import make_steerable_backend, run_pair


def eval_backend(name: str, backend, scenarios) -> dict:
    rows = []
    for s in scenarios:
        a_msg, b_alone, b_after = run_pair(backend, s)
        m = score_scenario(
            scenario_id=s.id,
            objective=s.objective,
            agent_a=a_msg,
            b_alone=b_alone,
            b_after=b_after,
            gold_answer=s.gold_answer,
            gold_answers=getattr(s, "gold_answers", ()) or (),
            score_mode=getattr(s, "score_mode", "text") or "text",
            mc_gold_letter=getattr(s, "mc_gold_letter", "") or "",
            mc_wrong_letter=getattr(s, "mc_wrong_letter", "") or "",
        )
        rows.append(
            {
                "scenario": asdict(s),
                "agent_a": a_msg,
                "b_alone": b_alone,
                "b_after": b_after,
                "metrics": asdict(m),
            }
        )
        print(
            f"  [{name}] {m.scenario_id}: influence={m.influence:+.3f} "
            f"success={m.success} ({m.notes})"
        )
    return {
        "backend": name,
        "n": len(rows),
        "n_success": sum(1 for r in rows if r["metrics"]["success"]),
        "mean_influence": sum(r["metrics"]["influence"] for r in rows) / max(1, len(rows)),
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", default="dry_run")
    p.add_argument("--steered", default="dry_run_steered")
    p.add_argument("--hf-model", default=None, help="HF id for alqr backend")
    p.add_argument("--model-key", default=None)
    p.add_argument("--pickle-dir", default=None)
    p.add_argument("--no-quant", action="store_true")
    p.add_argument(
        "--baseline-objective",
        default="none",
        choices=["auto", "resist", "accept", "none"],
        help="For alqr baseline: 'none' = unsteered generate",
    )
    p.add_argument(
        "--steered-objective",
        default="auto",
        choices=["auto", "resist", "accept", "none"],
    )
    p.add_argument(
        "--scenario-set",
        default="all",
        choices=["all", "no_country"],
        help="no_country drops Australia/Canada capital items",
    )
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "runs"),
    )
    args = p.parse_args()

    scenarios = get_scenarios(scenario_set=args.scenario_set)
    print("=" * 72)
    print(f"BASELINE: {args.baseline}  objective={args.baseline_objective}")
    base_backend = make_steerable_backend(
        args.baseline,
        model=args.hf_model,
        pickle_dir=args.pickle_dir,
        model_key=args.model_key,
        quant=not args.no_quant,
        objective_mode=args.baseline_objective,
    )
    baseline = eval_backend(args.baseline, base_backend, scenarios)

    print("=" * 72)
    print(f"STEERED: {args.steered}  objective={args.steered_objective}")
    steered_backend = make_steerable_backend(
        args.steered,
        model=args.hf_model,
        pickle_dir=args.pickle_dir,
        model_key=args.model_key,
        quant=not args.no_quant,
        objective_mode=args.steered_objective,
    )
    steered = eval_backend(args.steered, steered_backend, scenarios)

    delta = {
        "success_baseline": baseline["n_success"],
        "success_steered": steered["n_success"],
        "success_delta": steered["n_success"] - baseline["n_success"],
        "mean_influence_baseline": baseline["mean_influence"],
        "mean_influence_steered": steered["mean_influence"],
        "mean_influence_delta": steered["mean_influence"] - baseline["mean_influence"],
    }

    print("=" * 72)
    print(
        f"SUCCESS  {delta['success_baseline']}/{baseline['n']} → "
        f"{delta['success_steered']}/{steered['n']}  "
        f"(Δ {delta['success_delta']:+d})"
    )
    print(
        f"MEAN INFLUENCE  {delta['mean_influence_baseline']:+.3f} → "
        f"{delta['mean_influence_steered']:+.3f}  "
        f"(Δ {delta['mean_influence_delta']:+.3f})"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"steered_compare_{args.baseline}_vs_{args.steered}_{stamp}.json"
    payload = {
        "created_utc": stamp,
        "delta": delta,
        "baseline": baseline,
        "steered": steered,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
