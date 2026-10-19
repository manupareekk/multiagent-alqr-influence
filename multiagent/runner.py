#!/usr/bin/env python3
"""Run multi-agent influence measurement.

Phase 1 of the research plan: measure continuous influence BEFORE steering.

Examples:
  python -m multiagent.runner --backend dry_run
  python -m multiagent.runner --backend dry_run_calibrated
  OPENAI_API_KEY=... python -m multiagent.runner --backend openai
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .backends import (
    agent_a_system,
    agent_b_after_system,
    agent_b_alone_system,
    format_b_after_user,
    format_b_alone_user,
    make_backend,
)
from .metrics import score_scenario
from .scenarios import get_scenarios


def run(backend_name: str, model: str | None, out_dir: Path) -> Path:
    backend = make_backend(backend_name, model=model)
    scenarios = get_scenarios()
    rows = []

    print(f"backend={backend_name}  scenarios={len(scenarios)}")
    print("-" * 72)

    for s in scenarios:
        # Use scripted A message from scenario (controlled stimulus).
        # Optionally could generate A via backend; scripted is more reproducible.
        a_msg = s.agent_a_message

        b_alone = backend.generate(
            agent_b_alone_system(),
            format_b_alone_user(s),
        )
        b_after = backend.generate(
            agent_b_after_system(),
            format_b_after_user(s, a_msg),
        )

        result = score_scenario(
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

        row = {
            "scenario": asdict(s),
            "agent_a": a_msg,
            "b_alone": b_alone,
            "b_after": b_after,
            "metrics": asdict(result),
        }
        rows.append(row)

        m = result
        print(
            f"[{m.scenario_id}] obj={m.objective}  "
            f"influence={m.influence:+.3f}  "
            f"gold_alone={m.alone_has_gold} gold_after={m.after_has_gold}  "
            f"success={m.success}  ({m.notes})"
        )
        print(f"  B alone: {b_alone[:110]}")
        print(f"  B after: {b_after[:110]}")
        print()

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"influence_run_{backend_name}_{stamp}.json"
    payload = {
        "backend": backend_name,
        "model": model,
        "created_utc": stamp,
        "n": len(rows),
        "n_success": sum(1 for r in rows if r["metrics"]["success"]),
        "mean_influence": sum(r["metrics"]["influence"] for r in rows) / max(1, len(rows)),
        "rows": rows,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("-" * 72)
    print(
        f"saved {out_path}  "
        f"success={payload['n_success']}/{payload['n']}  "
        f"mean_influence={payload['mean_influence']:+.3f}"
    )
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-agent influence harness")
    p.add_argument(
        "--backend",
        default="dry_run",
        choices=["dry_run", "dry_run_calibrated", "openai"],
    )
    p.add_argument("--model", default=None, help="OpenAI model name if --backend openai")
    p.add_argument(
        "--out-dir",
        default=str(Path(__file__).resolve().parent / "runs"),
        help="Where to write JSON logs",
    )
    args = p.parse_args()
    run(args.backend, args.model, Path(args.out_dir))


if __name__ == "__main__":
    main()
