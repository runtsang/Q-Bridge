#!/usr/bin/env python3
"""Generate scaled feature map modules with gpt-oss-20b."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

BASE_DIR = Path(__file__).resolve().parent
SEED_ROOT = (
    BASE_DIR.parent
    / "seed_codebase"
    / "QML-Classification"
    / "feature_map"
)
PROMPT_PATH = BASE_DIR / "gpt_oss_20b_feature_map_prompt.txt"
PARADIGMS: Sequence[str] = ("extension", "controlled modification")
DEFAULT_RUN_NAME = "feature_map_runs1"

TARGET_QML_ROOT: Path
RAW_OUTPUT_ROOT: Path
INFERENCE_LOG_PATH: Path


def configure_output_paths(run_name: str) -> None:
    """Configure global output directories for a particular run."""

    global TARGET_QML_ROOT, RAW_OUTPUT_ROOT, INFERENCE_LOG_PATH

    run_root = BASE_DIR / run_name
    TARGET_QML_ROOT = run_root / "feature_maps"
    RAW_OUTPUT_ROOT = run_root / "raw_responses"
    INFERENCE_LOG_PATH = run_root / "inference_log.json"


configure_output_paths(DEFAULT_RUN_NAME)

STRUCTURED_CODE_PATTERN = re.compile(
    r"(?P<field>qml_code):\s*(?P<quote>'''|\"\"\")(?:\s*\n)?(?P<content>.*?)(?P=quote)",
    re.DOTALL,
)


@dataclass(frozen=True)
class GenerationContext:
    """Description of the current seed and requested scaling variant."""

    seed_path: Path
    paradigm: str
    suggested_output: Path


@dataclass(frozen=True)
class GenerationResult:
    """Container for model output and derived metadata."""

    raw_response: str
    qml_code: str
    summary: str
    name: Optional[str]
    scaling_paradigm: Optional[str]


def ensure_directories(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


def sanitize_generated_name(name: str) -> str:
    """Convert the model-proposed name into a safe Python module identifier."""

    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "scaled_feature_map"
    if cleaned[0].isdigit():
        cleaned = f"fm_{cleaned}"
    return cleaned


def append_inference_log(entry: Dict[str, Any]) -> None:
    """Append a JSON record describing a single generation."""

    ensure_directories(INFERENCE_LOG_PATH.parent)
    with INFERENCE_LOG_PATH.open("a", encoding="utf-8") as handle:
        json.dump(entry, handle)
        handle.write("\n")


@lru_cache()
def load_prompt_template() -> str:
    """Load and memoize the static prompt template."""

    if not PROMPT_PATH.exists():
        raise FileNotFoundError(f"Missing feature map prompt template at {PROMPT_PATH}.")
    return PROMPT_PATH.read_text(encoding="utf-8")


def list_seed_candidates() -> Sequence[Path]:
    """Return all available seed feature map modules."""

    if not SEED_ROOT.exists():
        raise FileNotFoundError(f"Seed directory {SEED_ROOT} does not exist.")
    return [path for path in sorted(SEED_ROOT.glob("*.py")) if path.name != "__init__.py"]


def select_reference(rng: random.Random, *, candidates: Sequence[Path]) -> Path:
    """Pick a random seed module from the candidate pool."""

    if not candidates:
        raise RuntimeError(f"No seed modules found in {SEED_ROOT}.")
    return rng.choice(candidates)


def select_paradigm(rng: random.Random) -> str:
    """Choose a scaling paradigm for the generation request."""

    return rng.choice(tuple(PARADIGMS))


def build_generation_context(
    *,
    reference: Path,
    paradigm: str,
    target_dir: Optional[str],
) -> GenerationContext:
    """Construct contextual metadata required for prompting and output."""

    if target_dir is not None:
        suggested_root = Path(target_dir)
    else:
        suggested_root = Path("feature_map_scaled")

    suggested_name = f"{reference.stem}_{paradigm.replace(' ', '_')}"
    suggested_output = suggested_root / f"{suggested_name}.py"
    return GenerationContext(reference, paradigm, suggested_output)


def build_reference_block(context: GenerationContext) -> str:
    """Render the automation block appended to the static prompt."""

    relative_seed = context.seed_path.relative_to(BASE_DIR.parent)
    code = context.seed_path.read_text(encoding="utf-8")

    block = (
        "#TargetFeatureMap\n"
        f"relative_output_path: {context.suggested_output.as_posix()}\n"
        f"seed_reference: {relative_seed.as_posix()}\n"
        f"scaling_paradigm: {context.paradigm}\n\n"
        "```python\n"
        "#ReferenceFeatureMap\n"
        f"seed_path: {relative_seed.as_posix()}\n\n"
        f"{code}\n"
        "```"
    )
    return block


def build_prompt(context: GenerationContext) -> str:
    """Insert the automation block into the static prompt template."""

    template = load_prompt_template()
    return template.replace("{{REFERENCE_BLOCK}}", build_reference_block(context))


def parse_structured_generation_payload(text: str, index) -> GenerationResult:
    """Parse the structured response emitted by the model."""

    normalized = text.replace("\r\n", "\n")
    anchor = normalized.lower().find("assistantfinal")
    if anchor != -1:
        normalized = normalized[anchor + len("assistantfinal") :]
    normalized = normalized.lstrip()

    code_fields: Dict[str, str] = {}
    for match in STRUCTURED_CODE_PATTERN.finditer(normalized):
        code_fields[match.group("field")] = match.group("content").strip()

    qml_code = code_fields.get("qml_code")
    if not qml_code:
        raise ValueError("Model response did not include a qml_code block.")

    def _match_last_line(pattern: str) -> Optional[str]:
        matches = list(re.finditer(pattern, normalized))
        if not matches:
            return None
        return matches[-1].group(1).strip()

    name = _match_last_line(r"\bname:\s*(.+)")
    scaling = _match_last_line(r"\bscaling_paradigm:\s*(.+)")

    summary_matches = list(
        re.finditer(
            r"\bsummary:\s*(.*?)(?=\n(?:qml_code|name|scaling_paradigm)\b|$)",
            normalized,
            re.DOTALL,
        )
    )
    summary_text = summary_matches[-1].group(1) if summary_matches else ""
    summary_lines = [line.strip() for line in summary_text.strip().splitlines() if line.strip()]
    summary = " ".join(summary_lines)

    return GenerationResult(
        raw_response=text,
        qml_code=qml_code,
        summary=summary,
        name=name + '-' + str(index),
        scaling_paradigm=scaling,
    )


_PIPELINE_CACHE: Dict[
    Tuple[str, str, Optional[Tuple[Tuple[str, str], ...]]],
    Any,
] = {}


def load_text_generation_pipeline(
    model_id: str,
    *,
    device_map: str,
    max_memory: Optional[Mapping[str, str]] = None,
):
    """Lazy loader around Hugging Face text-generation pipelines."""

    try:  # pragma: no cover - optional dependency
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        import torch
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "transformers (and torch) must be installed to run generation"
        ) from exc

    cache_key = (
        model_id,
        device_map,
        tuple(sorted(max_memory.items())) if max_memory else None,
    )
    if cache_key in _PIPELINE_CACHE:
        return _PIPELINE_CACHE[cache_key]

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    dtype: Optional["torch.dtype"] = None
    if torch.cuda.is_available():
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    model_kwargs = {
        "pretrained_model_name_or_path": model_id,
        "device_map": device_map,
    }
    if dtype is not None:
        model_kwargs["torch_dtype"] = dtype
    if max_memory is not None:
        model_kwargs["max_memory"] = dict(max_memory)

    model = AutoModelForCausalLM.from_pretrained(**model_kwargs)
    pipeline_kwargs = {
        "task": "text-generation",
        "model": model,
        "tokenizer": tokenizer,
    }

    if getattr(model, "hf_device_map", None) is None:
        pipeline_kwargs["device"] = 0 if torch.cuda.is_available() else -1

    text_generation_pipeline = pipeline(**pipeline_kwargs)
    _PIPELINE_CACHE[cache_key] = text_generation_pipeline
    return text_generation_pipeline


def call_hf_model(
    prompt: str,
    *,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    device_map: str,
    max_memory: Optional[Mapping[str, str]],
) -> str:
    """Invoke gpt-oss-20b (or a compatible model) using transformers."""

    pipe = load_text_generation_pipeline(
        model_id,
        device_map=device_map,
        max_memory=max_memory,
    )
    outputs = pipe(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
    )
    return outputs[0]["generated_text"]


def parse_max_gpu_memory(spec: Optional[str]) -> Optional[Dict[str, str]]:
    """Parse CLI max-memory overrides into the expected mapping."""

    if spec is None:
        return None
    entries = [entry.strip() for entry in spec.split(",") if entry.strip()]
    if not entries:
        return None
    parsed: Dict[str, str] = {}
    for index, entry in enumerate(entries):
        if "=" in entry:
            device_id, limit = entry.split("=", 1)
            device_id = device_id.strip()
            limit = limit.strip()
            if not device_id:
                raise ValueError("missing device identifier before '='")
        else:
            device_id = f"cuda:{index}"
            limit = entry.strip()
        if not limit:
            raise ValueError(f"missing memory limit for device {device_id!r}")
        if device_id.isdigit():
            device_id = f"cuda:{device_id}"
        parsed[device_id] = limit
    return parsed


def write_raw_response(relative_path: Path, response: str, *, suffix: str = "") -> None:
    """Persist the raw model response for auditing."""

    ensure_directories(RAW_OUTPUT_ROOT)
    sanitized = relative_path.as_posix().replace("/", "__")
    target = RAW_OUTPUT_ROOT / f"{sanitized}{suffix}.txt"
    target.write_text(response, encoding="utf-8")


def write_qml_module(relative_path: Path, code: str) -> None:
    """Write the generated feature map module to disk."""

    target = TARGET_QML_ROOT / relative_path
    ensure_directories(target.parent)
    target.write_text(code.rstrip() + "\n", encoding="utf-8")


def generate_feature_map(
    context: GenerationContext,
    prompt: str,
    *,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    device_map: str,
    max_memory: Optional[Mapping[str, str]],
    index: int,
) -> GenerationResult:
    """Run inference for a single feature map scaling task."""

    raw_response = call_hf_model(
        prompt,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device_map=device_map,
        max_memory=max_memory,
    )
    return parse_structured_generation_payload(raw_response, index=index)


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=os.environ.get("GPT_OSS_20B_MODEL", "openai/gpt-oss-20b"))
    parser.add_argument(
        "--device-map",
        default=os.environ.get("UNVERIFIABLE_DEVICE_MAP", "auto"),
        help="Device map strategy for AutoModelForCausalLM.from_pretrained",
    )
    parser.add_argument(
        "--max-gpu-memory",
        default=os.environ.get("UNVERIFIABLE_MAX_GPU_MEMORY"),
        help="Optional per-device memory limits for model loading.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=262144)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--num-outputs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility.")
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="Optional relative directory for suggested output modules.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=DEFAULT_RUN_NAME,
        help="Directory (relative to dataset/) where outputs will be saved.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Emit prompts without calling the model.",
    )
    args = parser.parse_args(argv)

    configure_output_paths(args.run_name)
    ensure_directories(TARGET_QML_ROOT, RAW_OUTPUT_ROOT, INFERENCE_LOG_PATH.parent)

    rng = random.Random(args.seed)
    candidates = list_seed_candidates()
    max_memory = parse_max_gpu_memory(args.max_gpu_memory)

    for index in range(1, args.num_outputs + 1):
        reference = select_reference(rng, candidates=candidates)
        paradigm = select_paradigm(rng)
        context = build_generation_context(
            reference=reference,
            paradigm=paradigm,
            target_dir=args.target_dir,
        )

        prompt = build_prompt(context)
        relative_seed = context.seed_path.relative_to(BASE_DIR.parent)

        if args.dry_run:
            write_raw_response(context.suggested_output, prompt, suffix=".prompt")
            print(
                f"[dry-run] wrote prompt for {context.suggested_output} "
                f"(seed: {relative_seed}, paradigm: {context.paradigm})"
            )
            continue

        try:
            result = generate_feature_map(
                context,
                prompt,
                model_id=args.model_id,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device_map=args.device_map,
                max_memory=max_memory,
                index=index,
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            write_raw_response(context.suggested_output, prompt, suffix=".prompt")
            print(
                f"Generation failed for {context.suggested_output}: {exc}",
                file=sys.stderr,
            )
            continue

        proposed_name = result.name or context.suggested_output.stem
        sanitized_name = sanitize_generated_name(proposed_name)
        final_relative_path = context.suggested_output.with_name(f"{sanitized_name}.py")

        write_raw_response(final_relative_path, result.raw_response)
        write_qml_module(final_relative_path, result.qml_code)

        append_inference_log(
            {
                "id": index,
                "seed_reference": relative_seed.as_posix(),
                "scaling_paradigm": result.scaling_paradigm or context.paradigm,
                "generated_name": sanitized_name,
                "raw_generated_name": proposed_name,
                "suggested_output": context.suggested_output.as_posix(),
                "final_output": final_relative_path.as_posix(),
                "summary": result.summary,
                "length": len(result.qml_code),
            }
        )

        print(
            "Generated feature map module "
            f"{final_relative_path.as_posix()} "
            f"from seed {relative_seed.as_posix()} (paradigm: {context.paradigm})"
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
