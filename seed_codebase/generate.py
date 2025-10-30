#!/usr/bin/env python3
"""Utility for generating scaled ML/QML file pairs with gpt-oss-20b."""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import sys
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple
from typing import Literal


SINGLE_REFERENCE_PROMPT = "gpt_oss_20b_scaling_prompt_single.txt"
MULTI_REFERENCE_PROMPT = "gpt_oss_20b_scaling_prompt_multi.txt"
BASE_DIR = Path(__file__).resolve().parent
PROMPT_PATH_SINGLE = BASE_DIR / SINGLE_REFERENCE_PROMPT
PROMPT_PATH_MULTI = BASE_DIR / MULTI_REFERENCE_PROMPT
SEED_ML_ROOT = BASE_DIR.parent / "seed_codebase" / "ML-Github"
SEED_QML_ROOT = BASE_DIR.parent / "seed_codebase" / "QML-Github"
DEFAULT_RUN_NAME = "un-1"


def configure_output_paths(run_name: str) -> None:
    """Update global output directories based on ``run_name``."""

    global TARGET_ML_ROOT, TARGET_QML_ROOT, RAW_OUTPUT_ROOT, INFERENCE_LOG_PATH

    run_root = BASE_DIR / Path(run_name)
    TARGET_ML_ROOT = run_root / "ML"
    TARGET_QML_ROOT = run_root / "QML"
    RAW_OUTPUT_ROOT = run_root / "raw_responses"
    INFERENCE_LOG_PATH = run_root / "inference_log.json"


configure_output_paths(DEFAULT_RUN_NAME)

ML_MARKER = "#<<ML>>"
QML_MARKER = "#<<QML>>"
CODE_FENCE = "```"
PYTHON_BLOCK_PATTERN = re.compile(
    r"```(?:python)?[ \t]*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
STRUCTURED_CODE_PATTERN = re.compile(
    r"(?P<field>ml_code|qml_code):\s*(?P<quote>'''|\"\"\")(?:\s*\n)?(?P<content>.*?)(?P=quote)",
    re.DOTALL,
)


@lru_cache()
def load_base_prompt(reference_count: int) -> str:
    """Return the static instruction block that matches the reference budget."""

    if reference_count <= 0:
        raise ValueError("reference_count must be positive")

    if reference_count == 1:
        prompt_path = PROMPT_PATH_SINGLE
    else:
        prompt_path = PROMPT_PATH_MULTI

    if not prompt_path.exists():
        raise FileNotFoundError(
            f"Missing prompt template at {prompt_path}."
        )
    return prompt_path.read_text(encoding="utf-8")


@dataclass(frozen=True)
class PairContext:
    """Metadata describing a matched ML/QML seed file."""

    relative_path: Path
    ml_seed_path: Path
    qml_seed_path: Path


    def render_context_block(
        self,
        include_code: bool = True,
        *,
        label: str = "#Pair",
    ) -> str:
        """Create the reference section appended to the base prompt."""
        header = (
            f"{label}\n"
            f"relative_path: {self.relative_path.as_posix()}\n"
            f"ml_seed_path: {self.ml_seed_path.as_posix()}\n"
            f"qml_seed_path: {self.qml_seed_path.as_posix()}"
        )

        if not include_code:
            return header + "\n"

        ml_code = self.ml_seed_path.read_text(encoding="utf-8")
        qml_code = self.qml_seed_path.read_text(encoding="utf-8")
        return (
            header
            + "\n\n```python\n"
            + ml_code
            + "\n```\n\n```python\n"
            + qml_code
            + "\n```\n"
        )


def iter_common_pairs(
    ml_root: Path,
    qml_root: Path,
    include: Optional[Sequence[str]] = None,
    exclude: Optional[Sequence[str]] = None,
) -> Iterator[PairContext]:
    """Yield seed file pairs sharing the same relative Python path."""

    include_set = {Path(p) for p in include or ()}
    exclude_set = {Path(p) for p in exclude or ()}

    for ml_path in sorted(ml_root.rglob("*.py")):
        rel = ml_path.relative_to(ml_root)
        if include_set and rel not in include_set:
            continue
        if rel in exclude_set:
            continue
        qml_path = qml_root / rel
        if qml_path.exists():
            yield PairContext(rel, ml_path, qml_path)


def ensure_directories(*dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class GenerationTask:
    """Description of a single scaling request."""

    references: Sequence[PairContext]
    target_relative_path: Path
    anchor_relative_path: Path
    generation_index: int


@dataclass(frozen=True)
class GenerationResult:
    """Container for model outputs and derived metadata."""

    raw_response: str
    ml_code: str
    qml_code: str
    summary: str
    name: Optional[str] = None
    scaling_paradigm: Optional[str] = None


def sanitize_generated_name(name: str) -> str:
    """Normalize the model-proposed name into a safe module identifier."""

    cleaned = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        cleaned = "scaled_module"
    if cleaned[0].isdigit():
        cleaned = f"mod_{cleaned}"
    return cleaned


def append_inference_log(entry: dict) -> None:
    """Append a single inference record to the JSONL audit log."""

    ensure_directories(INFERENCE_LOG_PATH.parent)
    with INFERENCE_LOG_PATH.open("a", encoding="utf-8") as handle:
        json.dump(entry, handle)
        handle.write("\n")


def parse_structured_generation_payload(text: str) -> Optional[Dict[str, str]]:
    """Extract structured fields emitted by newer prompts.

    Some generations (see ``Autoencoder_gen001.py.txt``) emit a plain text block
    that resembles::

        assistantfinal
        name: Example
        scaling_paradigm: combination
        summary: ...
        ml_code: '''...'''
        qml_code: '''...'''

    This helper pulls out the ``ml_code``/``qml_code`` sections along with the
    metadata so downstream tooling can persist the resulting modules.
    """

    normalized = text.replace("\r\n", "\n")
    lower_normalized = normalized.lower()
    anchor = lower_normalized.find("assistantfinal")
    if anchor != -1:
        normalized = normalized[anchor + len("assistantfinal") :]
    normalized = normalized.lstrip()

    # Extract code blocks.
    code_fields: Dict[str, str] = {}
    for match in STRUCTURED_CODE_PATTERN.finditer(normalized):
        field = match.group("field")
        content = match.group("content")
        code_fields[field] = content.strip()

    if not {"ml_code", "qml_code"}.issubset(code_fields):
        return None

    def _match_line(pattern: str) -> Optional[str]:
        found = re.search(pattern, normalized)
        if not found:
            return None
        return found.group(1).strip()


    def _match_last_line(pattern: str) -> Optional[str]:
        matches = list(re.finditer(pattern, normalized))
        if not matches:
            return None
        return matches[-1].group(1).strip()

    name = _match_last_line(r"\bname:\s*(.+)")
    # Use the last occurrence if multiple are present
    scaling = _match_last_line(r"\bscaling_paradigm:\s*(.+)")

    summary_matches = list(
        re.finditer(
            r"\bsummary:\s*(.*?)(?=\n(?:ml_code|name|scaling_paradigm)\b|$)",
            normalized,
            re.DOTALL,
        )
    )
    summary_text = summary_matches[-1].group(1) if summary_matches else ""
    summary_lines = [line.strip() for line in summary_text.strip().splitlines() if line.strip()]
    summary = " ".join(summary_lines)

    payload: Dict[str, str] = {
        "ml_code": code_fields["ml_code"],
        "qml_code": code_fields["qml_code"],
        "summary": summary,
    }
    if name:
        payload["name"] = name
    if scaling:
        payload["scaling_paradigm"] = scaling
    return payload


def parse_reference_probabilities(spec: Optional[str]) -> Optional[Tuple[float, ...]]:
    """Parse user-provided sampling weights for reference counts 1-5."""

    if spec is None:
        return None
    parts = [part.strip() for part in spec.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("must provide five comma-separated values for counts 1-4.")
    weights = tuple(float(part) for part in parts)
    if any(weight < 0 for weight in weights):
        raise ValueError("probabilities must be non-negative.")
    if sum(weights) <= 0:
        raise ValueError("probabilities must sum to a positive value.")
    return weights


def parse_max_gpu_memory(spec: Optional[str]) -> Optional[Dict[str, str]]:
    """Parse CLI max-memory overrides into the Hugging Face expected mapping."""

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


def select_reference_count(
    rng: random.Random, reference_cap: int, weights: Optional[Sequence[float]]
) -> int:
    """Sample how many reference pairs to supply, respecting optional weights."""

    available_counts = list(range(1, reference_cap + 1))
    if weights is None:
        return rng.choice(available_counts)
    truncated = [weights[count - 1] for count in available_counts]
    total = sum(truncated)
    if total <= 0:
        raise ValueError("Probability mass for available reference counts is zero.")
    return rng.choices(available_counts, weights=truncated, k=1)[0]


def build_prompt(task: GenerationTask, base_prompt: str, *, backend: str) -> str:
    """Attach the generation metadata and seed references to the base prompt."""

    # The backend argument is kept for interface stability, but prompt templates
    # now encode all formatting/scaling guidance directly.
    _ = backend

    blocks = [
        base_prompt.rstrip(),
        "",
        (
            "#TargetOutput\n"
            f"relative_path: {task.target_relative_path.as_posix()}\n"
            f"anchor_reference: {task.anchor_relative_path.as_posix()}\n"
            f"reference_pair_count: {len(task.references)}"
        ),
    ]

    for idx, context in enumerate(task.references, start=1):
        blocks.extend(
            [
                "",
                context.render_context_block(label=f"#ReferencePair[{idx}]").strip(),
            ]
        )

    return "\n".join(blocks) + "\n"


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
    """Lazy loader around Hugging Face pipelines for gpt-oss-20b."""
    try:
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

    # Prefer bfloat16 weights on accelerators that support it to avoid dtype
    # mismatches introduced by torch/Accelerate when mixing precision modes.
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

    # When the model is loaded with `device_map="auto"`, Accelerate manages the
    # device placement. In that scenario the pipeline API raises an error if we
    # attempt to manually override the device. Only set the explicit device when
    # the model was not partitioned by Accelerate (e.g. CPU-only execution).
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
    """Invoke gpt-oss-20b (or a drop-in replacement) using transformers."""
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


@lru_cache()
def load_openai_client():
    """Lazily instantiate the OpenAI client when structured outputs are used."""

    try:  # pragma: no cover - optional dependency
        from openai import OpenAI
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "openai must be installed to request structured generations"
        ) from exc

    return OpenAI()


def call_openai_structured(
    prompt: str,
    *,
    model: str,
    temperature: float,
    max_output_tokens: int,
) -> GenerationResult:
    """Invoke the OpenAI Responses API and parse structured generation payloads."""

    try:  # pragma: no cover - optional dependency
        from pydantic import BaseModel, Field
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "pydantic must be installed to parse structured generations"
        ) from exc

    class GenerationStructuredOutput(BaseModel):
        name: str = Field(
            description=(
                "Module/Class name to use for both generated files."
            )
        )
        scaling_paradigm: Literal[
            "extension", "combination", "controlled modification"
        ] = Field(
            description=(
                "Primary scaling paradigm guiding the redesign."
            )
        )
        summary: str = Field(
            description=(
                "2-3 short sentences of major upgrades shared across the ML/QML outputs."
            )
        )
        ml_code: str = Field(
            description="Python source code for the scaled classical ML implementation."
        )
        qml_code: str = Field(
            description="Python source code for the scaled quantum ML implementation."
        )

    client = load_openai_client()
    response = client.responses.parse(
        model=model,
        input=[
            {
                "role": "system",
                "content": (
                    "You are gpt-oss-20b, an expert hybrid quantum-classical ML engineer "
                    "tasked with scaling repository assets."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        text_format=GenerationStructuredOutput,
    )

    parsed = response.output_parsed
    return GenerationResult(
        raw_response=response.model_dump_json(indent=2),
        ml_code=parsed.ml_code,
        qml_code=parsed.qml_code,
        summary=parsed.summary,
        name=parsed.name,
        scaling_paradigm=parsed.scaling_paradigm,
    )


def generate_outputs(
    prompt: str,
    *,
    task: GenerationTask,
    backend: str,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    openai_model: str,
    openai_max_output_tokens: int,
    device_map: str,
    max_memory: Optional[Mapping[str, str]],
    generation_index: int,
) -> GenerationResult:
    """Dispatch to the requested backend and normalize the resulting payloads."""

    if backend == "openai":
        result = call_openai_structured(
            prompt,
            model=openai_model,
            temperature=temperature,
            max_output_tokens=openai_max_output_tokens,
        )
        return result

    if backend != "hf":
        raise ValueError(f"Unsupported backend: {backend}")

    response = call_hf_model(
        prompt,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        device_map=device_map,
        max_memory=max_memory,
    )
    structured_payload = parse_structured_generation_payload(response)
    if structured_payload:
        return GenerationResult(
            raw_response=response,
            ml_code=structured_payload["ml_code"],
            qml_code=structured_payload["qml_code"],
            summary=structured_payload.get("summary", ""),
            name=structured_payload.get("name") + '-' + str(generation_index),
            scaling_paradigm=structured_payload.get("scaling_paradigm"),
        )

    ml_code = extract_code_section(response, ML_MARKER, fallback_index=0)
    qml_code = extract_code_section(response, QML_MARKER, fallback_index=1)
    return GenerationResult(
        raw_response=response,
        ml_code=ml_code,
        qml_code=qml_code,
        summary="",
    )


def extract_code_section(
    text: str,
    marker: str,
    *,
    fallback_index: Optional[int] = None,
) -> str:
    """Return the fenced code block that follows ``marker``.

    Older prompts instructed the model to emit explicit ``#<<ML>>``/``#<<QML>>``
    anchors before each code fence. Some generations occasionally omit those
    markers, which previously caused the entire run to abort. To make the
    dataset tooling more robust we fall back to positional extraction (first
    python block for ML, second for QML) when the explicit marker is absent.
    """

    marker_idx = text.find(marker)
    if marker_idx == -1:
        if fallback_index is not None:
            matches = PYTHON_BLOCK_PATTERN.findall(text)
            if 0 <= fallback_index < len(matches):
                return matches[fallback_index]
        raise ValueError(f"Missing marker {marker!r} in model response.")
    fence_start = text.find(CODE_FENCE, marker_idx)
    if fence_start == -1:
        raise ValueError(f"Missing opening code fence after {marker!r}.")
    fence_start += len(CODE_FENCE)
    lang_terminator = text.find("\n", fence_start)
    if lang_terminator == -1:
        raise ValueError("Malformed code fence (missing newline).")
    code_start = lang_terminator + 1
    fence_end = text.find(CODE_FENCE, code_start)
    if fence_end == -1:
        raise ValueError(f"Missing closing code fence for marker {marker!r}.")
    return text[code_start:fence_end]


def dump_outputs(
    relative_path: Path,
    ml_code: str,
    qml_code: str,
    *,
    base_dir_ml: Optional[Path] = None,
    base_dir_qml: Optional[Path] = None,
) -> None:
    ml_root = base_dir_ml or TARGET_ML_ROOT
    qml_root = base_dir_qml or TARGET_QML_ROOT
    ml_target = ml_root / relative_path
    qml_target = qml_root / relative_path
    ensure_directories(ml_target.parent, qml_target.parent)
    ml_target.write_text(ml_code.rstrip() + "\n", encoding="utf-8")
    qml_target.write_text(qml_code.rstrip() + "\n", encoding="utf-8")


def write_raw_response(
    relative_path: Path,
    response: str,
    *,
    suffix: str = "",
    extension: str = ".md",
) -> None:
    ensure_directories(RAW_OUTPUT_ROOT)
    sanitized = relative_path.as_posix().replace("/", "__")
    target = RAW_OUTPUT_ROOT / f"{sanitized}{suffix}{extension}"
    target.write_text(response, encoding="utf-8")


def process_generation(
    task: GenerationTask,
    base_prompt: str,
    *,
    model_id: str,
    max_new_tokens: int,
    temperature: float,
    dry_run: bool,
    backend: str,
    openai_model: str,
    openai_max_output_tokens: int,
    device_map: str,
    max_memory: Optional[Mapping[str, str]],
    generation_index: int,
) -> None:
    prompt = build_prompt(task, base_prompt, backend=backend)
    if dry_run:
        ensure_directories(RAW_OUTPUT_ROOT)
        write_raw_response(
            task.target_relative_path,
            prompt,
            suffix=".prompt",
            extension=".txt",
        )
        print(
            f"[dry-run] wrote prompt for {task.target_relative_path} (anchor: "
            f"{task.anchor_relative_path}) using {len(task.references)} reference pair(s)"
        )
        return

    result = generate_outputs(
        prompt,
        task=task,
        backend=backend,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        openai_model=openai_model,
        openai_max_output_tokens=openai_max_output_tokens,
        device_map=device_map,
        max_memory=max_memory,
        generation_index=generation_index,
    )
    raw_extension = "txt"
    proposed_name = result.name or task.target_relative_path.stem
    sanitized_name = sanitize_generated_name(proposed_name)
    final_relative_path = Path(f"{sanitized_name}.py")
    write_raw_response(
        final_relative_path,
        result.raw_response,
        extension=raw_extension,
    )
    dump_outputs(
        relative_path=final_relative_path,
        ml_code=result.ml_code,
        qml_code=result.qml_code,
    )
    print(
        "Generated scaled pair for "
        f"{final_relative_path} (anchor: {task.anchor_relative_path}) "
        f"using {len(task.references)} reference pair(s)"
    )
    ml_length = len(result.ml_code)
    qml_length = len(result.qml_code)
    average_length = (ml_length + qml_length) / 2
    references_info = []
    for idx, ref in enumerate(task.references):
        alias = string.ascii_lowercase[idx]
        references_info.append(
            {
                "index": idx + 1,
                "alias": alias,
                "relative_path": ref.relative_path.as_posix()
            }
        )
    append_inference_log(
        {
            "id": task.generation_index,
            "reference_number": len(task.references),
            "references": references_info,
            "generated_name": sanitized_name,
            "raw_generated_name": proposed_name,
            "scaling_paradigm": result.scaling_paradigm,
            "summary": result.summary,
            "lengths": {
                "ml": ml_length,
                "qml": qml_length,
                "average": average_length,
            },
        }
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend",
        choices=("hf", "openai"),
        default="hf",
        help="Generation backend: 'hf' for Hugging Face transformers or 'openai' for the Responses API.",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("GPT_OSS_20B_MODEL", "openai/gpt-oss-20b"),
        help="Hugging Face model identifier or local path for gpt-oss-20b.",
    )
    parser.add_argument(
        "--device-map",
        default=os.environ.get("UNVERIFIABLE_DEVICE_MAP", "auto"),
        help=(
            "Device map strategy supplied to AutoModelForCausalLM.from_pretrained "
            "when using the Hugging Face backend (e.g., 'auto', 'balanced')."
        ),
    )
    parser.add_argument(
        "--max-gpu-memory",
        default=os.environ.get("UNVERIFIABLE_MAX_GPU_MEMORY"),
        help=(
            "Optional per-device memory limits for model loading. Provide as "
            "comma-separated values like '20GiB,20GiB,20GiB,20GiB' or explicit "
            "mappings such as 'cuda:0=24GiB,cuda:1=24GiB'."
        ),
    )
    parser.add_argument(
        "--openai-model",
        default=os.environ.get("OPENAI_RESPONSES_MODEL", "gpt-4.1-mini"),
        help="OpenAI Responses API model to use when --backend=openai.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=262144,
        help="Maximum number of new tokens to sample per file pair.",
    )
    parser.add_argument(
        "--openai-max-output-tokens",
        type=int,
        default=int(os.environ.get("OPENAI_MAX_OUTPUT_TOKENS", "4096")),
        help="Maximum number of output tokens when using the OpenAI backend.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature supplied to the text-generation pipeline.",
    )
    parser.add_argument(
        "--num-outputs",
        type=int,
        default=1000,
        help="Number of scaled file pairs to generate (looping with random references).",
    )
    parser.add_argument(
        "--reference-probabilities",
        default=None,
        help=(
            "Comma-separated probabilities for selecting 1-5 reference pairs "
            "(e.g., '0.4,0.3,0.2,0.05,0.05'). Defaults to a uniform 20% each."
        ),
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional seed for the random sampler controlling reference selection.",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        default=None,
        help="Subset of relative paths to process (space separated).",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Relative paths to skip (space separated).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write prompts to disk without calling the language model.",
    )
    parser.add_argument(
        "--run-name",
        default=DEFAULT_RUN_NAME,
        help=(
            "Directory name under the dataset output root used to store generated "
            "ML/QML pairs and metadata."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    configure_output_paths(args.run_name)
    # Ensure the output directories exist before generation begins.
    ensure_directories(TARGET_ML_ROOT, TARGET_QML_ROOT, RAW_OUTPUT_ROOT)


    # Discover seed pairs that share matching relative paths across ML/QML roots.
    pairs = list(
        iter_common_pairs(
            SEED_ML_ROOT,
            SEED_QML_ROOT,
            include=args.include,
            exclude=args.exclude,
        )
    )
    if not pairs:
        print("No common seed pairs were discovered.")
        return 0

    if args.num_outputs < 1:
        print("num-outputs must be at least 1 to trigger generation.")
        return 0

    # Configure randomness and weighting for selecting reference context per output.
    rng = random.Random(args.random_seed)
    try:
        reference_weights = parse_reference_probabilities(
            args.reference_probabilities
        )
    except ValueError as exc:
        print(f"Invalid --reference-probabilities: {exc}", file=sys.stderr)
        return 1
    if reference_weights is None:
        reference_weights = (0.1, 0.0, 0.0, 0.0)
    try:
        max_memory = parse_max_gpu_memory(args.max_gpu_memory)
    except ValueError as exc:
        print(f"Invalid --max-gpu-memory: {exc}", file=sys.stderr)
        return 1
    reference_cap = min(4, len(pairs))
    if reference_cap == 0:
        print("Insufficient seed pairs available for reference sampling.")
        return 0

    start_time = time.perf_counter()
    for generation_index in range(1, args.num_outputs + 1):
        try:
            reference_count = select_reference_count(
                rng, reference_cap, reference_weights
            )
        except ValueError as exc:
            print(
                f"Invalid reference sampling configuration: {exc}",
                file=sys.stderr,
            )
            return 1
        references = rng.sample(pairs, reference_count)
        primary_context = references[0]
        target_relative_path = primary_context.relative_path.with_name(
            f"{primary_context.relative_path.stem}__gen{generation_index:03d}"
            f"{primary_context.relative_path.suffix}"
        )
        task = GenerationTask(
            references=references,
            target_relative_path=target_relative_path,
            anchor_relative_path=primary_context.relative_path,
            generation_index=generation_index,
        )
        reference_paths = ", ".join(ref.relative_path.as_posix() for ref in references)
        elapsed = time.perf_counter() - start_time
        print(
            f"[{elapsed:.2f}s] [plan] Generation {generation_index}: "
            f"target={target_relative_path.as_posix()} "
            f"anchor={primary_context.relative_path.as_posix()} "
            f"references=[{reference_paths}]"
        )
        try:
            base_prompt = load_base_prompt(len(references))
        except Exception as exc:  # pragma: no cover - defensive logging
            print(
                f"[error] Failed to load prompt for {target_relative_path}: {exc}",
                file=sys.stderr,
            )
            continue
        try:
            process_generation(
                task,
                base_prompt,
                model_id=args.model_id,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                dry_run=args.dry_run,
                backend=args.backend,
                openai_model=args.openai_model,
                openai_max_output_tokens=args.openai_max_output_tokens,
                device_map=args.device_map,
                max_memory=max_memory,
                generation_index=task.generation_index,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            print(
                f"[error] Failed to process {task.target_relative_path}: {exc}",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    import torch
    print(torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(i, torch.cuda.get_device_name(i))
    raise SystemExit(main())