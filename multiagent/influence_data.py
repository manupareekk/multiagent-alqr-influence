#!/usr/bin/env python3
"""Build A-LQR pickles for multi-agent influence objectives.

Creates (under --pickle-dir):
  {key}-resist.pkl      # activations: resist bad peer
  {key}-defer.pkl       # activations: defer to bad peer
  {key}-accept.pkl      # activations: accept good critique
  {key}-ignore.pkl      # activations: ignore good critique
  {key}-influence_jac.pkl

Contrastive vectors used at steer time:
  X_resist = resist - defer     # push toward independence under bad A
  X_accept = accept - ignore    # push toward updating under good A

Requires CUDA + HF model access (run on Kaggle/PACE, not Mac CPU).

Example:
  python -m multiagent.influence_data --model gemma2b \\
      --pickle-dir /kaggle/working/pickles --quant
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["gemma2b", "gemma2bit", "qwen3b", "llama8b", "llama1b"],
        required=True,
    )
    parser.add_argument(
        "--pickle-dir",
        type=str,
        default=str(Path.cwd() / "pickles"),
        help="Where to write .pkl files (also updates steer config path)",
    )
    parser.add_argument("--quant", action="store_true", help="4-bit load")
    parser.add_argument("--n-contrast", type=int, default=None,
                        help="Cap samples per class (default: all available, max 200)")
    parser.add_argument("--n-jac", type=int, default=50)
    parser.add_argument(
        "--prompt-source",
        choices=["handmade", "tqa"],
        default="tqa",
        help="tqa = TruthfulQA-backed influence prompts (recommended)",
    )
    parser.add_argument(
        "--chat-template",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Wrap contrastive prompts in tokenizer chat template "
        "(default: on for gemma2bit, off otherwise)",
    )
    args = parser.parse_args()

    models = {
        "gemma2b": ("google/gemma-2-2b", "gemma-2-2b"),
        "gemma2bit": ("google/gemma-2-2b-it", "gemma-2-2b-it"),
        "qwen3b": ("Qwen/Qwen2.5-3B", "Qwen2.5-3B"),
        "llama8b": ("meta-llama/Meta-Llama-3-8B", "Llama-3-8B"),
        "llama1b": ("meta-llama/Llama-3.2-1B", "Llama-3.2-1B"),
    }
    model_name, key = models[args.model]
    pickle_dir = Path(args.pickle_dir).resolve()
    pickle_dir.mkdir(parents=True, exist_ok=True)

    # Point lab data_handling at our pickle dir before importing ContrastiveBuilder.
    cfg_path = Path(__file__).resolve().parents[1] / "steer" / "config" / "config.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        "environment:\n"
        f'  pickle_jar: "{pickle_dir}/"\n'
        '  ref_data_path: "./ref_data/"\n'
        '  tqa_data_path: "./tqa_data/"\n'
        '  tox_data_path: "./tox_data/"\n'
        '  con_data_path: "./con_data/"\n',
        encoding="utf-8",
    )
    print(f"wrote config pickle_jar -> {pickle_dir}")

    # Late imports so config is already patched.
    import torch as th
    from steer.data_handling import ContrastiveBuilder
    from steer.toxicity.tox_data_script import load_model

    from .influence_prompts import (
        accept_prompts,
        all_jac_prompts,
        apply_chat_template,
        defer_prompts,
        ignore_prompts,
        resist_prompts,
    )

    use_chat = (
        args.chat_template
        if args.chat_template is not None
        else args.model in ("gemma2bit",)
    )

    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(
        f"device={device}  model={model_name}  prompts={args.prompt_source}  "
        f"chat_template={use_chat}"
    )
    if device.type != "cuda":
        print(
            "WARNING: no CUDA. Jacobian collection will be extremely slow/OOM on CPU.",
            file=sys.stderr,
        )

    model, tokenizer = load_model(model_name, quant=args.quant)
    builder = ContrastiveBuilder(model, tokenizer)
    src = args.prompt_source

    def _prep(prompts: list[str]) -> list[str]:
        return apply_chat_template(prompts, tokenizer) if use_chat else prompts

    pairs = [
        ("resist", _prep(resist_prompts(src))),
        ("defer", _prep(defer_prompts(src))),
        ("accept", _prep(accept_prompts(src))),
        ("ignore", _prep(ignore_prompts(src))),
    ]

    for tag, prompts in pairs:
        n = len(prompts)
        if args.n_contrast is not None:
            n = min(n, args.n_contrast)
        n = min(n, 200)
        if n < 1:
            raise RuntimeError(f"no prompts for {tag}")
        fname = f"{key}-{tag}"
        print(f"collecting {fname}  n={n}")
        if use_chat and prompts:
            print(f"  chat-template sample[:120]: {prompts[0][:120]!r}")
        builder.collect_data_batch(prompts, n, fname)
        print(f"done with {fname}")

    jac_prompts = _prep(all_jac_prompts(src))
    n_jac = min(args.n_jac, len(jac_prompts), 50)
    jac_name = f"{key}-influence_jac"
    print(f"collecting jacobians {jac_name}  n={n_jac}")
    builder.collect_jacobians(jac_prompts, n_jac, jac_name, max_ctx=64)
    print(f"done with {jac_name}")
    print(f"pickles in {pickle_dir}")


if __name__ == "__main__":
    main()
