"""Agent-B backends with optional A-LQR steering.

Modes:
  - dry_run / dry_run_calibrated / openai: unsteered (from backends.py)
  - dry_run_steered: simulates successful influence steering (local test)
  - alqr: real LQRSteering on HF model (needs CUDA + influence pickles)
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .backends import (
    DryRunBackend,
    OpenAIBackend,
    agent_b_after_system,
    agent_b_alone_system,
    format_b_after_user,
    format_b_alone_user,
    make_backend,
)
from .scenarios import Scenario

ObjectiveMode = Literal["auto", "resist", "accept", "none"]


@dataclass
class DryRunSteeredBackend:
    """Uncalibrated alone; calibrated after A — mimics A-LQR fixing over/under influence."""

    def generate(self, system: str, user: str) -> str:
        alone = "WITHOUT seeing any other agent" in user
        if alone:
            return DryRunBackend(mode="uncalibrated").generate(system, user)
        return DryRunBackend(mode="calibrated").generate(system, user)


class ALQRSteeredBackend:
    """HuggingFace model + A-LQR on Agent B (especially B_after)."""

    def __init__(
        self,
        model_name: str,
        pickle_dir: str | Path,
        key: str,
        quant: bool = True,
        lmbda_resist: float = 1.25,
        lmbda_accept: float = 1.25,
        q: float = 0.1,
        r: float = 3.0,
        qf: float = 0.1,
        max_new_tokens: int = 80,
        objective_mode: ObjectiveMode = "auto",
        model=None,
        tokenizer=None,
        use_chat_template: bool = True,
    ):
        import torch as th
        from steer.steering import LQRSteering

        self.th = th
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.pickle_dir = Path(pickle_dir)
        self.key = key
        self.lmbda_resist = lmbda_resist
        self.lmbda_accept = lmbda_accept
        self.max_new_tokens = max_new_tokens
        self.objective_mode = objective_mode
        self.use_chat_template = use_chat_template

        if model is not None and tokenizer is not None:
            self.model, self.tokenizer = model, tokenizer
        else:
            from steer.toxicity.tox_data_script import load_model

            self.model, self.tokenizer = load_model(model_name, quant=quant)
            if hasattr(self.model, "to") and self.device.type == "cuda":
                try:
                    self.model = self.model.to(self.device)
                except Exception:
                    pass
        if getattr(self.tokenizer, "pad_token", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        resist = self._load(f"{key}-resist")
        defer = self._load(f"{key}-defer")
        accept = self._load(f"{key}-accept")
        ignore = self._load(f"{key}-ignore")
        jac = self._load(f"{key}-influence_jac")

        self.X_resist = (resist["X"] - defer["X"]).to(self.device)
        self.X_accept = (accept["X"] - ignore["X"]).to(self.device)
        self.A = jac["A"].to(self.device)

        # Default steerer; vectors swapped per call.
        self._LQRSteering = LQRSteering
        self.q, self.r, self.qf = q, r, qf
        self._steer_resist = LQRSteering(
            self.model,
            self.tokenizer,
            q=q,
            r=r,
            qf=qf,
            A=self.A.clone(),
            contrastive_vecs=self.X_resist,
            preserve_mem=True,
        )
        # Second controller needs its own K from same A — rebuild with accept vecs.
        # A was deleted inside first ctor if preserve_mem; reload jac for second.
        jac2 = self._load(f"{key}-influence_jac")
        A2 = jac2["A"].to(self.device)
        self._steer_accept = LQRSteering(
            self.model,
            self.tokenizer,
            q=q,
            r=r,
            qf=qf,
            A=A2,
            contrastive_vecs=self.X_accept,
            preserve_mem=True,
        )

    def _load(self, name: str) -> dict:
        path = self.pickle_dir / f"{name}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run: python -m multiagent.influence_data --model ..."
            )
        with open(path, "rb") as f:
            return pickle.load(f)

    def generate(self, system: str, user: str) -> str:
        # Alone: unsteered generate (or light resist if unsafe alone — keep simple).
        alone = "WITHOUT seeing any other agent" in user
        prompt = self._build_prompt(system, user)
        if alone or self.objective_mode == "none":
            return self._generate_plain(prompt)

        mode = self.objective_mode
        if mode == "auto":
            mode = self._infer_mode(user)

        steerer = self._steer_resist if mode == "resist" else self._steer_accept
        lmbda = self.lmbda_resist if mode == "resist" else self.lmbda_accept
        out = steerer.track_setpoint(
            [prompt],
            max_new_tokens=self.max_new_tokens,
            lmbda=lmbda,
            do_sample=False,
        )
        text = out[0]
        # Strip echoed prompt if model repeats it.
        if text.startswith(prompt):
            text = text[len(prompt) :].strip()
        return text.strip()

    def _build_prompt(self, system: str, user: str) -> str:
        """Prefer chat template for instruct models (e.g. gemma-2-2b-it)."""
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            # Gemma-IT has no system role — fold instructions into the user turn.
            content = f"{system}\n\n{user}"
            messages = [{"role": "user", "content": content}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return f"{system}\n\n{user}"

    def _infer_mode(self, user: str) -> Literal["resist", "accept"]:
        u = user.lower()
        # Prefer explicit markers from our scenario formatting.
        if "please correct to:" in u or "your draft may be wrong" in u:
            return "accept"
        if "ignore other possibilities" in u or "i'm confident the answer is:" in u:
            return "resist"
        # Legacy handmade cues
        resist_cues = [
            "pipe bomb",
            "malware",
            "steal",
            "sydney",
            "313",
            "don't refuse",
            "just answer",
            "i'm sure",
        ]
        accept_cues = [
            "that's wrong",
            "please correct",
            "correction",
            "mary shelley",
            "120+48",
            "not michelangelo",
        ]
        if any(c in u for c in accept_cues):
            return "accept"
        if any(c in u for c in resist_cues):
            return "resist"
        # Default: resist peer pressure (safer for multi-agent safety work).
        return "resist"

    def _generate_plain(self, prompt: str) -> str:
        th = self.th
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[-1]
        with th.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][prompt_len:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

class HFChatBackend:
    """Plain HF generate with chat template (no A-LQR / no pickles)."""

    def __init__(
        self,
        model,
        tokenizer,
        max_new_tokens: int = 80,
        use_chat_template: bool = True,
    ):
        import torch as th

        self.th = th
        self.model = model
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.use_chat_template = use_chat_template
        self.device = next(model.parameters()).device
        if getattr(tokenizer, "pad_token", None) is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    def _build_prompt(self, system: str, user: str) -> str:
        if self.use_chat_template and hasattr(self.tokenizer, "apply_chat_template"):
            content = f"{system}\n\n{user}"
            messages = [{"role": "user", "content": content}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                pass
        return f"{system}\n\n{user}"

    def generate(self, system: str, user: str) -> str:
        prompt = self._build_prompt(system, user)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_len = inputs["input_ids"].shape[-1]
        with self.th.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # Decode ONLY new tokens (chat templates break startswith(prompt) stripping).
        gen = out[0][prompt_len:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def make_steerable_backend(
    name: str,
    model: str | None = None,
    pickle_dir: str | None = None,
    model_key: str | None = None,
    quant: bool = True,
    objective_mode: ObjectiveMode = "auto",
    lmbda_resist: float = 1.25,
    lmbda_accept: float = 1.25,
    model_obj=None,
    tokenizer=None,
    use_chat_template: bool = True,
    max_new_tokens: int = 80,
):
    if name == "dry_run_steered":
        return DryRunSteeredBackend()
    if name in ("dry_run", "dry_run_calibrated", "openai"):
        return make_backend(name, model=model)
    if name == "hf_chat":
        if model_obj is None or tokenizer is None:
            raise ValueError("hf_chat requires model_obj and tokenizer")
        return HFChatBackend(
            model_obj,
            tokenizer,
            use_chat_template=use_chat_template,
            max_new_tokens=max_new_tokens,
        )
    if name == "alqr":
        if not pickle_dir or not model:
            raise ValueError("alqr backend requires --hf-model and --pickle-dir")
        # Map HF id → pickle key if not provided
        key = model_key or {
            "google/gemma-2-2b": "gemma-2-2b",
            "google/gemma-2-2b-it": "gemma-2-2b-it",
            "Qwen/Qwen2.5-3B": "Qwen2.5-3B",
            "meta-llama/Llama-3.2-1B": "Llama-3.2-1B",
            "meta-llama/Meta-Llama-3-8B": "Llama-3-8B",
        }.get(model)
        if not key:
            raise ValueError(f"Provide --model-key for unknown model {model}")
        return ALQRSteeredBackend(
            model_name=model,
            pickle_dir=pickle_dir,
            key=key,
            quant=quant,
            objective_mode=objective_mode,
            lmbda_resist=lmbda_resist,
            lmbda_accept=lmbda_accept,
            model=model_obj,
            tokenizer=tokenizer,
            use_chat_template=use_chat_template,
            max_new_tokens=max_new_tokens,
        )
    raise ValueError(f"Unknown backend: {name}")


def run_pair(backend, scenario: Scenario) -> tuple[str, str, str]:
    """Generate B alone + B after A. If backend is ALQR with objective_mode=auto,
    steer using the scenario's true objective (not text heuristics).
    """
    a_msg = scenario.agent_a_message
    prev = getattr(backend, "objective_mode", None)
    if prev == "auto" and hasattr(backend, "objective_mode"):
        backend.objective_mode = (
            "resist"
            if scenario.objective == "resist_overinfluence"
            else "accept"
        )
    try:
        b_alone = backend.generate(
            agent_b_alone_system(),
            format_b_alone_user(scenario),
        )
        b_after = backend.generate(
            agent_b_after_system(),
            format_b_after_user(scenario, a_msg),
        )
    finally:
        if prev is not None and hasattr(backend, "objective_mode"):
            backend.objective_mode = prev
    return a_msg, b_alone, b_after
