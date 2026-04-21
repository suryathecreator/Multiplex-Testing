#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_IMPORT_PATHS = [
    REPO_ROOT / "sglang-0.4.9.post6",
    REPO_ROOT / "transformers-4.54.0" / "src",
]
for import_path in LOCAL_IMPORT_PATHS:
    if import_path.exists():
        sys.path.insert(0, str(import_path))

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


AIME_PROMPT_TEMPLATE = """Solve the following AIME 2024 problem.

Reason carefully, then give the final answer in \\boxed{{}}.

Problem:
{problem}
"""


DEFAULT_PASS_AT_KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
DEFAULT_METHODS = (
    "baseline_independent",
    "shared_trace_branch_after_prefix",
    "standard_generation_independent",
)
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 30000
DEFAULT_API_KEY = "multiplex-local"
DEFAULT_DP_SIZE = 2
DEFAULT_TP_SIZE = 1
DEFAULT_REQUEST_BATCH_SIZE = 16
LEGACY_REQUEST_BATCH_SIZE = 64
LOW_MEMORY_REQUEST_BATCH_SIZE = 8
LOW_MEMORY_THRESHOLD_GB = 24.0
LOW_GPU_MAX_RUNNING_REQUESTS = 32
LOW_GPU_CHUNKED_PREFILL_SIZE = 2048
DEFAULT_CAPACITY_OF_STR_LEN = 32768
DEFAULT_MAX_NEW_TOKENS = 8192
DEFAULT_TIMEOUT_SECONDS = 3600
DEFAULT_CHECKPOINT_MATCHED_PROMPTS_STEP = 5
DEFAULT_REASONING_PREFIX_TOKEN_VALUES = (512, 1024, 2048)
RESOURCE_PROFILE_AUTO = "auto"
RESOURCE_PROFILE_TWO_GPU_SAFE = "two_gpu_safe"
RESOURCE_PROFILE_LEGACY_8GPU = "legacy_8gpu"
RESOURCE_PROFILE_CHOICES = (
    RESOURCE_PROFILE_AUTO,
    RESOURCE_PROFILE_TWO_GPU_SAFE,
    RESOURCE_PROFILE_LEGACY_8GPU,
)
NVCC_AUTO = object()
LOCALHOST_NO_PROXY_ENTRIES = ("127.0.0.1", "localhost", "::1")
# The fixed-prefix experiments do not rely on a literal opening think tag, so we
# intentionally avoid manually pre-filling one and let the model use its native
# chat template behavior.
ASSISTANT_THINK_PREFILL = ""
THINK_END_TAG = "</think>"
FIXED_PREFIX_NEVER_SWITCH_THINK_END = "<|fixed_prefix_never_switch|>"
MULTIPLEX_REQUIRED_ATTENTION_BACKENDS = {"flashinfer", "triton", "torch_native"}


@dataclass
class Example:
    prompt_index: int
    problem_id: str
    problem: str
    answer: str
    prompt_ids: List[int]
    assistant_prefill: str
    prompt_build_mode: str


@dataclass
class ParentTrace:
    benchmark: str
    prompt_index: int
    problem_id: str
    session_id: str
    parent_rid: str
    target_dp_rank: int
    success: bool
    message: str
    shared_trace_text: str
    branch_input_ids: List[int]
    cacheable_input_ids: List[int]
    uncached_tail_input_ids: List[int]
    cacheable_token_count: int
    eot_token_id: Optional[int]
    eot_output_index: int
    prompt_token_count: int
    response_token_count: int
    completion_tokens: int
    finish_reason: Any
    verification: Dict[str, Any]
    usable_for_eval: bool
    excluded_reason: Optional[str] = None
    generated_suffix: Optional[str] = None
    attempts_used: int = 0
    total_completion_tokens_spent: int = 0
    total_latency_seconds_spent: float = 0.0
    reject_reason_counts: Dict[str, int] | None = None
    accepted_attempt_index: Optional[int] = None
    reasoning_prefix_tokens: int = 0


@dataclass
class BaselinePromptResult:
    benchmark: str
    prompt_index: int
    problem_id: str
    target_dp_rank: int
    success: bool
    message: str
    required_sample_count: int
    usable_sample_count: int
    attempts_used: int
    total_completion_tokens_spent: int
    total_latency_seconds_spent: float
    reject_reason_counts: Dict[str, int]
    slot_statuses: List[Dict[str, Any]]
    reasoning_prefix_tokens: int = 0
    method: str = "baseline_independent"


@dataclass
class AttemptRecord:
    benchmark: str
    method: str
    prompt_index: int
    problem_id: str
    target_kind: str
    slot_index: Optional[int]
    attempt_index: int
    rid: str
    target_dp_rank: int
    accepted_for_eval: bool
    reject_reason: Optional[str]
    finish_reason: Any
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    latency_seconds: float
    planned_seed: Optional[int]
    contains_think_end: bool
    cumulative_attempts: int
    cumulative_completion_tokens: int
    budget_exhausted: bool
    terminal: bool
    generated_suffix: Optional[str] = None


@dataclass
class SampleRecord:
    benchmark: str
    method: str
    prompt_index: int
    problem_id: str
    sample_index: int
    rid: str
    target_dp_rank: int
    correct: bool
    score: float
    text: str
    finish_reason: Any
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    latency_seconds: float
    prefix_completion_tokens: int = 0
    session_id: Optional[str] = None
    parent_rid: Optional[str] = None
    parent_completion_tokens: int = 0
    cacheable_token_count: int = 0
    cache_verification_passed: Optional[bool] = None
    planned_seed: Optional[int] = None
    usable_for_eval: bool = True
    excluded_reason: Optional[str] = None
    score_reason: Optional[str] = None
    extracted_answer: Optional[str] = None
    score_debug: Optional[Dict[str, Any]] = None
    generated_suffix: Optional[str] = None
    accepted_attempt_index: Optional[int] = None
    reasoning_prefix_tokens: int = 0


@dataclass
class PromptBuildInfo:
    assistant_prefill: str
    prompt_build_mode: str


@dataclass
class ScoreResult:
    score: float
    reason: str
    extracted_answer: Optional[str]
    debug: Dict[str, Any]


@dataclass
class MultiplexSelfCheckResult:
    success: bool
    message: str
    attention_backend: Optional[str]
    enable_soft_thinking: Optional[bool]
    has_topk_metadata: bool
    finish_reason: Any
    output_text: str


@dataclass
class RuntimeConfig:
    requested_dp_size: int
    effective_dp_size: int
    requested_tp_size: int
    effective_tp_size: int
    requested_request_batch_size: Optional[int]
    request_batch_size: int
    visible_gpu_count: int
    per_gpu_memory_gb: Optional[float]
    requested_resource_profile: str
    resource_profile: str
    safe_request_batch_cap: Optional[int]
    cuda_graph_max_bs: Optional[int]
    chunked_prefill_size: Optional[int]
    max_running_requests: Optional[int]
    nvcc_path: Optional[str]
    attention_backend: Optional[str]
    decode_attention_backend: Optional[str]
    prefill_attention_backend: Optional[str]
    sampling_backend: Optional[str]
    disable_cuda_graph: bool
    disable_radix_cache: bool
    clamp_messages: List[str]


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: Dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=True) + "\n")


class StructuredEventLogger:
    def __init__(self, path: Path):
        self.writer = JsonlWriter(path)

    def log(self, tag: str, **payload: Any) -> None:
        record = {"ts": time.time(), "tag": tag, **payload}
        self.writer.append(record)


def ensure_localhost_no_proxy_env() -> None:
    for env_name in ("NO_PROXY", "no_proxy"):
        current_value = os.environ.get(env_name, "")
        entries = [item.strip() for item in current_value.split(",") if item.strip()]
        existing = set(entries)
        for host in LOCALHOST_NO_PROXY_ENTRIES:
            if host not in existing:
                entries.append(host)
        os.environ[env_name] = ",".join(entries)


def build_local_requests_session(requests_module: Any) -> Any:
    session = requests_module.Session()
    session.trust_env = False
    return session


class SGLangRestClient:
    def __init__(self, base_url: str, api_key: Optional[str], timeout: int):
        import requests

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._requests = requests
        self.session = build_local_requests_session(requests)
        headers = {"Content-Type": "application/json; charset=utf-8"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.session.headers.update(headers)

    def close(self) -> None:
        self.session.close()

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Any:
        response = self.session.post(
            f"{self.base_url}{endpoint}",
            json=payload,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"{endpoint} failed with status {response.status_code}: {response.text}"
            )
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    def _get(self, endpoint: str) -> Any:
        response = self.session.get(
            f"{self.base_url}{endpoint}",
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(
                f"{endpoint} failed with status {response.status_code}: {response.text}"
            )
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    def open_session(self, capacity_of_str_len: int, session_id: Optional[str] = None) -> str:
        payload = {"capacity_of_str_len": capacity_of_str_len}
        if session_id is not None:
            payload["session_id"] = session_id
        result = self._post("/open_session", payload)
        if not isinstance(result, str):
            raise RuntimeError(f"Unexpected /open_session response: {result!r}")
        return result

    def close_session(self, session_id: str) -> None:
        self._post("/close_session", {"session_id": session_id})

    def fork_request(
        self,
        session_id: str,
        parent_rid: str,
        child_count: int,
        child_rids: List[str],
        child_seeds: List[int],
        target_dp_rank: int,
        allow_non_eot_branch: bool = False,
    ) -> Dict[str, Any]:
        return self._post(
            "/fork_request",
            {
                "session_id": session_id,
                "parent_rid": parent_rid,
                "child_count": child_count,
                "child_rids": child_rids,
                "child_seeds": child_seeds,
                "target_dp_rank": target_dp_rank,
                "allow_non_eot_branch": allow_non_eot_branch,
            },
        )

    def generate(
        self,
        *,
        input_ids: Optional[List[int] | List[List[int]]] = None,
        text: Optional[str | List[str]] = None,
        sampling_params: Dict[str, Any] | List[Dict[str, Any]],
        rid: Optional[str | List[str]] = None,
        session_params: Optional[Dict[str, Any] | List[Dict[str, Any]]] = None,
        data_parallel_rank: Optional[int] = None,
        return_logprob: bool = False,
    ) -> Any:
        payload: Dict[str, Any] = {
            "sampling_params": sampling_params,
            "return_logprob": return_logprob,
        }
        if input_ids is not None:
            payload["input_ids"] = input_ids
        if text is not None:
            payload["text"] = text
        if rid is not None:
            payload["rid"] = rid
        if session_params is not None:
            payload["session_params"] = session_params
        if data_parallel_rank is not None:
            payload["data_parallel_rank"] = data_parallel_rank
        return self._post("/generate", payload)

    def get_server_info(self) -> Dict[str, Any]:
        result = self._get("/get_server_info")
        if not isinstance(result, dict):
            raise RuntimeError(f"Unexpected /get_server_info response: {result!r}")
        return result


class SGLangServerHandle:
    def __init__(self, server_args: Any, timeout: int):
        self.server_args = server_args
        self.timeout = timeout
        self.process: Optional[mp.Process] = None

    def start(self) -> None:
        from sglang.srt.entrypoints.http_server import launch_server

        self.process = mp.Process(target=launch_server, args=(self.server_args,))
        self.process.start()
        self._wait_healthy()

    def stop(self) -> None:
        from sglang.srt.utils import kill_process_tree

        if self.process is None:
            return
        if self.process.is_alive():
            kill_process_tree(self.process.pid)
            self.process.join(timeout=30)
        self.process = None

    def _wait_healthy(self) -> None:
        import requests

        base_url = self.server_args.url()
        headers = {}
        if self.server_args.api_key:
            headers["Authorization"] = f"Bearer {self.server_args.api_key}"

        start = time.time()
        with build_local_requests_session(requests) as session:
            while time.time() - start < self.timeout:
                if self.process is not None and not self.process.is_alive():
                    raise RuntimeError("SGLang server process exited during startup.")
                try:
                    response = session.get(
                        f"{base_url}/health_generate",
                        headers=headers,
                        timeout=30,
                    )
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
            else:
                raise TimeoutError("Timed out waiting for the SGLang server to become healthy.")

            while time.time() - start < self.timeout:
                if self.process is not None and not self.process.is_alive():
                    raise RuntimeError("SGLang server process exited during cache warmup.")
                try:
                    response = session.get(
                        f"{base_url}/flush_cache",
                        headers=headers,
                        timeout=30,
                    )
                    if response.status_code == 200:
                        return
                except requests.RequestException:
                    pass
                time.sleep(2)
        raise TimeoutError("Timed out waiting for the SGLang server cache to become ready.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs shared-trace pass@k on AIME 2024.")
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--benchmark", default="aime2024", choices=["aime2024"])
    parser.add_argument("--max-k", type=int, default=16)
    parser.add_argument(
        "--methods",
        default="baseline,shared_trace,standard_generation",
        help="Comma-separated subset of baseline,shared_trace,standard_generation",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--api-key", default=DEFAULT_API_KEY)
    parser.add_argument("--dp-size", type=int, default=DEFAULT_DP_SIZE)
    parser.add_argument("--tp-size", type=int, default=DEFAULT_TP_SIZE)
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=None,
        help="Per-request chunk size. When omitted, the resource profile chooses a safe default.",
    )
    parser.add_argument(
        "--resource-profile",
        default=RESOURCE_PROFILE_AUTO,
        choices=RESOURCE_PROFILE_CHOICES,
    )
    parser.add_argument("--capacity-of-str-len", type=int, default=DEFAULT_CAPACITY_OF_STR_LEN)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--server-timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-prompts", type=int, default=50)
    parser.add_argument(
        "--reasoning-prefix-token-values",
        default=",".join(str(value) for value in DEFAULT_REASONING_PREFIX_TOKEN_VALUES),
        help="Comma-separated fixed reasoning-prefix token counts to evaluate, e.g. 512,1024,2048.",
    )
    parser.add_argument(
        "--checkpoint-matched-prompts-step",
        type=int,
        default=DEFAULT_CHECKPOINT_MATCHED_PROMPTS_STEP,
        help="Write summary checkpoints after every N prompts where both methods have full usable k data.",
    )
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    return parser.parse_args()


def parse_reasoning_prefix_token_values(raw_values: str) -> List[int]:
    values: List[int] = []
    seen: set[int] = set()
    for item in raw_values.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"Reasoning prefix tokens must be positive integers, got {value}.")
        if value in seen:
            continue
        values.append(value)
        seen.add(value)
    if not values:
        raise ValueError("At least one reasoning prefix token value must be provided.")
    return values


def configure_logging(output_dir: Path, logger_name: str = "compare_passk_aime") -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(output_dir / "run.log")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def count_visible_gpus_from_env(cuda_visible_devices: Optional[str]) -> Optional[int]:
    if cuda_visible_devices is None:
        return None
    value = cuda_visible_devices.strip()
    if not value:
        return 0
    if value.lower() in {"none", "void", "n/a", "-1"}:
        return 0
    return len([item for item in value.split(",") if item.strip()])


def detect_visible_gpu_count() -> int:
    env_count = count_visible_gpus_from_env(os.environ.get("CUDA_VISIBLE_DEVICES"))
    if env_count is not None:
        return env_count
    try:
        import torch
    except Exception:
        return 0
    try:
        return int(torch.cuda.device_count())
    except Exception:
        return 0


def detect_per_gpu_memory_gb(visible_gpu_count: int) -> Optional[float]:
    if visible_gpu_count <= 0:
        return None
    try:
        import torch
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        inspect_count = min(int(torch.cuda.device_count()), visible_gpu_count)
        if inspect_count <= 0:
            return None
        memories = [
            torch.cuda.get_device_properties(index).total_memory / (1024 ** 3)
            for index in range(inspect_count)
        ]
    except Exception:
        return None
    if not memories:
        return None
    return round(min(memories), 2)


def detect_nvcc_path() -> Optional[str]:
    candidate_paths: List[Path] = []

    cudacxx = os.environ.get("CUDACXX")
    if cudacxx:
        candidate_paths.append(Path(cudacxx).expanduser())

    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        cuda_home = os.environ.get(env_var)
        if cuda_home:
            candidate_paths.append(Path(cuda_home).expanduser() / "bin" / "nvcc")

    which_nvcc = shutil.which("nvcc")
    if which_nvcc:
        candidate_paths.append(Path(which_nvcc))

    candidate_paths.append(Path("/usr/local/cuda/bin/nvcc"))

    seen: set[str] = set()
    for candidate in candidate_paths:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        try:
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate_str
        except OSError:
            continue
    return None


def select_resource_profile(requested_profile: str, effective_dp_size: int) -> str:
    if requested_profile != RESOURCE_PROFILE_AUTO:
        return requested_profile
    if effective_dp_size <= 2:
        return RESOURCE_PROFILE_TWO_GPU_SAFE
    return RESOURCE_PROFILE_LEGACY_8GPU


def safe_request_batch_cap(per_gpu_memory_gb: Optional[float]) -> int:
    if per_gpu_memory_gb is not None and per_gpu_memory_gb <= LOW_MEMORY_THRESHOLD_GB:
        return LOW_MEMORY_REQUEST_BATCH_SIZE
    return DEFAULT_REQUEST_BATCH_SIZE


def resolve_runtime_config(
    args: argparse.Namespace,
    *,
    visible_gpu_count: Optional[int] = None,
    per_gpu_memory_gb: Optional[float] = None,
    nvcc_path: Any = NVCC_AUTO,
) -> RuntimeConfig:
    requested_dp_size = int(args.dp_size)
    requested_tp_size = int(args.tp_size)
    requested_request_batch_size = (
        None if args.request_batch_size is None else int(args.request_batch_size)
    )

    if requested_dp_size < 1:
        raise ValueError("--dp-size must be >= 1")
    if requested_tp_size < 1:
        raise ValueError("--tp-size must be >= 1")
    if requested_request_batch_size is not None and requested_request_batch_size < 1:
        raise ValueError("--request-batch-size must be >= 1 when provided")

    visible_gpu_count = (
        detect_visible_gpu_count() if visible_gpu_count is None else int(visible_gpu_count)
    )
    if visible_gpu_count < 1:
        raise RuntimeError(
            "No visible GPUs detected. Check CUDA_VISIBLE_DEVICES or the Slurm GPU allocation."
        )

    if per_gpu_memory_gb is None:
        per_gpu_memory_gb = detect_per_gpu_memory_gb(visible_gpu_count)

    if nvcc_path is NVCC_AUTO:
        nvcc_path = detect_nvcc_path()
    if nvcc_path is not None:
        nvcc_path = str(nvcc_path)

    clamp_messages: List[str] = []
    effective_dp_size = min(requested_dp_size, visible_gpu_count)
    if effective_dp_size != requested_dp_size:
        clamp_messages.append(
            f"requested dp_size={requested_dp_size} but only {visible_gpu_count} visible GPUs were detected; using dp_size={effective_dp_size}"
        )

    effective_tp_size = requested_tp_size
    if requested_tp_size != 1:
        effective_tp_size = 1
        clamp_messages.append(
            f"requested tp_size={requested_tp_size} is not supported by the 1.5B eval profile; using tp_size=1"
        )

    resource_profile = select_resource_profile(args.resource_profile, effective_dp_size)
    safe_cap: Optional[int] = None
    if resource_profile == RESOURCE_PROFILE_TWO_GPU_SAFE:
        safe_cap = safe_request_batch_cap(per_gpu_memory_gb)
        request_batch_size = (
            safe_cap
            if requested_request_batch_size is None
            else requested_request_batch_size
        )
        if request_batch_size > safe_cap:
            clamp_messages.append(
                f"requested request_batch_size={request_batch_size} exceeds the {resource_profile} safety cap of {safe_cap}; using {safe_cap}"
            )
            request_batch_size = safe_cap
        cuda_graph_max_bs = request_batch_size
        chunked_prefill_size = LOW_GPU_CHUNKED_PREFILL_SIZE
        max_running_requests = LOW_GPU_MAX_RUNNING_REQUESTS
    else:
        request_batch_size = (
            LEGACY_REQUEST_BATCH_SIZE
            if requested_request_batch_size is None
            else requested_request_batch_size
        )
        cuda_graph_max_bs = None
        chunked_prefill_size = None
        max_running_requests = None

    attention_backend: Optional[str] = None
    decode_attention_backend: Optional[str] = None
    prefill_attention_backend: Optional[str] = None
    sampling_backend: Optional[str] = None
    disable_cuda_graph = False
    disable_radix_cache = False
    if not nvcc_path:
        disable_cuda_graph = True
        cuda_graph_max_bs = None
        if per_gpu_memory_gb is not None and per_gpu_memory_gb <= LOW_MEMORY_THRESHOLD_GB:
            attention_backend = "torch_native"
            decode_attention_backend = "torch_native"
            prefill_attention_backend = "torch_native"
            sampling_backend = "pytorch"
            chunked_prefill_size = -1
            max_running_requests = request_batch_size
            disable_radix_cache = True
            clamp_messages.append(
                "nvcc was not found on this low-memory node; falling back to torch_native attention with pytorch sampling, disabling CUDA graph, disabling chunked prefill, and disabling radix cache for a slower but more robust real multiplex path"
            )
        else:
            attention_backend = "triton"
            decode_attention_backend = "triton"
            prefill_attention_backend = "triton"
            clamp_messages.append(
                "nvcc was not found on this node; forcing Triton attention and disabling CUDA graph so the run stays on the real multiplex path without FlashInfer JIT compilation"
            )

    return RuntimeConfig(
        requested_dp_size=requested_dp_size,
        effective_dp_size=effective_dp_size,
        requested_tp_size=requested_tp_size,
        effective_tp_size=effective_tp_size,
        requested_request_batch_size=requested_request_batch_size,
        request_batch_size=request_batch_size,
        visible_gpu_count=visible_gpu_count,
        per_gpu_memory_gb=per_gpu_memory_gb,
        requested_resource_profile=args.resource_profile,
        resource_profile=resource_profile,
        safe_request_batch_cap=safe_cap,
        cuda_graph_max_bs=cuda_graph_max_bs,
        chunked_prefill_size=chunked_prefill_size,
        max_running_requests=max_running_requests,
        nvcc_path=nvcc_path,
        attention_backend=attention_backend,
        decode_attention_backend=decode_attention_backend,
        prefill_attention_backend=prefill_attention_backend,
        sampling_backend=sampling_backend,
        disable_cuda_graph=disable_cuda_graph,
        disable_radix_cache=disable_radix_cache,
        clamp_messages=clamp_messages,
    )


def log_runtime_config(logger: logging.Logger, runtime_config: RuntimeConfig) -> None:
    for message in runtime_config.clamp_messages:
        logger.warning("[setup] %s", message)
    logger.info(
        "[setup] runtime config: visible_gpus=%s per_gpu_memory_gb=%s requested_dp_size=%s effective_dp_size=%s requested_tp_size=%s effective_tp_size=%s requested_request_batch_size=%s request_batch_size=%s resource_profile=%s cuda_graph_max_bs=%s chunked_prefill_size=%s max_running_requests=%s nvcc_path=%s attention_backend=%s sampling_backend=%s disable_cuda_graph=%s disable_radix_cache=%s",
        runtime_config.visible_gpu_count,
        runtime_config.per_gpu_memory_gb,
        runtime_config.requested_dp_size,
        runtime_config.effective_dp_size,
        runtime_config.requested_tp_size,
        runtime_config.effective_tp_size,
        runtime_config.requested_request_batch_size,
        runtime_config.request_batch_size,
        runtime_config.resource_profile,
        runtime_config.cuda_graph_max_bs,
        runtime_config.chunked_prefill_size,
        runtime_config.max_running_requests,
        runtime_config.nvcc_path,
        runtime_config.attention_backend or "auto",
        runtime_config.sampling_backend or "auto",
        runtime_config.disable_cuda_graph,
        runtime_config.disable_radix_cache,
    )


def build_pass_at_k_values(max_k: int) -> List[int]:
    values = [value for value in DEFAULT_PASS_AT_KS if value <= max_k]
    if max_k not in values:
        values.append(max_k)
    return values


def build_prompt(problem: str) -> str:
    return AIME_PROMPT_TEMPLATE.format(problem=problem.strip())


def reconstruct_assistant_text(prefix: str, generated_suffix: str) -> str:
    if generated_suffix.startswith(prefix):
        return generated_suffix
    return prefix + generated_suffix


def build_prefilled_prompt_ids(
    tokenizer: Any,
    prompt_text: str,
    assistant_prefill: str = ASSISTANT_THINK_PREFILL,
) -> Tuple[List[int], PromptBuildInfo]:
    messages = [{"role": "user", "content": prompt_text}]

    try:
        prompt_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        return prompt_ids, PromptBuildInfo(
            assistant_prefill=assistant_prefill,
            prompt_build_mode="native_generation_prompt",
        )
    except Exception:
        templated_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(templated_text, add_special_tokens=False)
        return prompt_ids, PromptBuildInfo(
            assistant_prefill=assistant_prefill,
            prompt_build_mode="native_generation_prompt_text",
        )


def normalize_methods(raw_methods: str) -> List[str]:
    mapping = {
        "baseline": "baseline_independent",
        "baseline_independent": "baseline_independent",
        "shared_trace": "shared_trace_branch_after_prefix",
        "shared_trace_branch_after_prefix": "shared_trace_branch_after_prefix",
        "standard": "standard_generation_independent",
        "standard_generation": "standard_generation_independent",
        "standard_generation_independent": "standard_generation_independent",
    }
    methods = []
    for item in raw_methods.split(","):
        item = item.strip()
        if not item:
            continue
        if item not in mapping:
            raise ValueError(f"Unknown method {item!r}.")
        methods.append(mapping[item])
    if not methods:
        raise ValueError("At least one method must be selected.")
    return methods


def dataset_row_keys(row: Any) -> List[str]:
    keys = getattr(row, "keys", None)
    if callable(keys):
        try:
            return [str(key) for key in keys()]
        except Exception:
            return []
    return []


def dataset_row_value(
    row: Any,
    candidate_keys: Iterable[str],
    default: Any = None,
) -> Any:
    sentinel = object()
    for key in candidate_keys:
        try:
            if key in row:
                value = row[key]
                if value is not None:
                    return value
        except Exception:
            pass

        getter = getattr(row, "get", None)
        if callable(getter):
            try:
                value = getter(key, sentinel)
            except Exception:
                value = sentinel
            if value is not sentinel and value is not None:
                return value
    return default


def normalize_problem_id(row: Any, prompt_index: int) -> str:
    raw_problem_id = dataset_row_value(
        row,
        ("ID", "id", "problem_id", "problemId", "uid"),
        default=prompt_index,
    )
    if raw_problem_id is None:
        raw_problem_id = prompt_index
    problem_id = str(raw_problem_id).strip()
    return problem_id or str(prompt_index)


def normalize_required_text_field(
    row: Any,
    prompt_index: int,
    field_label: str,
    candidate_keys: Iterable[str],
) -> str:
    value = dataset_row_value(row, candidate_keys, default=None)
    if value is None:
        available_keys = dataset_row_keys(row)
        raise ValueError(
            f"AIME row {prompt_index} is missing the {field_label} field. "
            f"Checked keys={list(candidate_keys)!r}. Available keys={available_keys!r}."
        )

    text_value = str(value).strip()
    if not text_value:
        available_keys = dataset_row_keys(row)
        raise ValueError(
            f"AIME row {prompt_index} has an empty {field_label} field after normalization. "
            f"Checked keys={list(candidate_keys)!r}. Available keys={available_keys!r}."
        )
    return text_value


def load_examples(
    tokenizer: Any,
    benchmark: str,
    max_prompts: Optional[int],
) -> Tuple[List[Example], PromptBuildInfo]:
    if benchmark != "aime2024":
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    from datasets import load_dataset

    dataset = load_dataset("Maxwell-Jia/AIME_2024", split="train")
    examples: List[Example] = []
    prompt_build_info: Optional[PromptBuildInfo] = None
    for idx, row in enumerate(dataset):
        if max_prompts is not None and idx >= max_prompts:
            break
        problem_text = normalize_required_text_field(
            row,
            idx,
            "problem",
            ("Problem", "problem", "question", "Question"),
        )
        answer_text = normalize_required_text_field(
            row,
            idx,
            "answer",
            ("Answer", "answer", "solution", "target"),
        )
        prompt_text = build_prompt(problem_text)
        prompt_ids, row_prompt_build_info = build_prefilled_prompt_ids(tokenizer, prompt_text)
        if prompt_build_info is None:
            prompt_build_info = row_prompt_build_info
        examples.append(
            Example(
                prompt_index=idx,
                problem_id=normalize_problem_id(row, idx),
                problem=problem_text,
                answer=answer_text,
                prompt_ids=prompt_ids,
                assistant_prefill=row_prompt_build_info.assistant_prefill,
                prompt_build_mode=row_prompt_build_info.prompt_build_mode,
            )
        )
    if prompt_build_info is None:
        prompt_build_info = PromptBuildInfo(
            assistant_prefill=ASSISTANT_THINK_PREFILL,
            prompt_build_mode="native_generation_prompt",
        )
    return examples, prompt_build_info


def make_server_args(args: argparse.Namespace, runtime_config: RuntimeConfig) -> Any:
    from sglang.srt.server_args import ServerArgs

    return ServerArgs(
        model_path=args.model,
        tokenizer_path=args.model,
        served_model_name=args.model,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        dp_size=runtime_config.effective_dp_size,
        tp_size=runtime_config.effective_tp_size,
        pp_size=1,
        random_seed=args.seed,
        mem_fraction_static=args.mem_fraction_static,
        attention_backend=runtime_config.attention_backend,
        decode_attention_backend=runtime_config.decode_attention_backend,
        prefill_attention_backend=runtime_config.prefill_attention_backend,
        sampling_backend=runtime_config.sampling_backend,
        cuda_graph_max_bs=runtime_config.cuda_graph_max_bs,
        disable_cuda_graph=runtime_config.disable_cuda_graph,
        chunked_prefill_size=runtime_config.chunked_prefill_size,
        max_running_requests=runtime_config.max_running_requests,
        disable_radix_cache=runtime_config.disable_radix_cache,
        # Multiplex/soft-thinking maintains per-request top-k state across decode
        # steps. The overlap scheduler can race ahead with placeholder/future
        # tokens before that state is fully materialized, which has been the root
        # cause of several malformed top-k crashes on 2-GPU no-nvcc nodes.
        # Prefer correctness and fidelity over throughput here.
        disable_overlap_schedule=True,
        # This experiment does not use constrained/structured decoding, so
        # explicitly disable grammar backends instead of inheriting SGLang's
        # optional xgrammar default.
        grammar_backend="none",
        reasoning_parser="qwen3",
        enable_soft_thinking=True,
        think_end_str="</think>",
        max_topk=3,
        used_topk=3,
        enable_max_topk=False,
        after_thinking_temperature=0.6,
        after_thinking_top_p=0.95,
        after_thinking_top_k=-1,
        after_thinking_min_p=0.0,
    )


def runtime_config_payload(runtime_config: RuntimeConfig) -> Dict[str, Any]:
    return asdict(runtime_config)


def serialize_server_args(server_args: Any) -> Dict[str, Any]:
    if hasattr(server_args, "__dataclass_fields__"):
        return asdict(server_args)
    return dict(vars(server_args))


def strip_markdown_noise(text: str) -> str:
    cleaned = text.replace("```latex", "```").replace("```text", "```")
    cleaned = re.sub(r"```(?:[^`\n]*)\n", "", cleaned)
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.replace("`", "")
    return cleaned


def normalize_candidate_text(text: str) -> str:
    cleaned = strip_markdown_noise(text)
    cleaned = cleaned.replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{2,}", "\n", cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.strip(" \n\t,.;:")
    return cleaned


def isoformat_utc(unix_ts: Optional[float]) -> Optional[str]:
    if unix_ts is None:
        return None
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).isoformat()


def format_wall_clock_seconds(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def build_run_timing(
    started_at_unix: float,
    finished_at_unix: Optional[float] = None,
) -> Dict[str, Any]:
    run_timing = {
        "started_at_unix": started_at_unix,
        "started_at": isoformat_utc(started_at_unix),
    }
    if finished_at_unix is not None:
        wall_clock_seconds = max(finished_at_unix - started_at_unix, 0.0)
        run_timing.update(
            {
                "finished_at_unix": finished_at_unix,
                "finished_at": isoformat_utc(finished_at_unix),
                "wall_clock_seconds": wall_clock_seconds,
                "wall_clock_hms": format_wall_clock_seconds(wall_clock_seconds),
            }
        )
    return run_timing


def extract_last_boxed_answer(text: str) -> Optional[str]:
    start = text.rfind("\\boxed{")
    while start != -1:
        cursor = start + len("\\boxed{")
        depth = 1
        while cursor < len(text):
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start + len("\\boxed{") : cursor]
            cursor += 1
        start = text.rfind("\\boxed{", 0, start)
    return None


def extract_tail_after_final_think(text: str) -> str:
    if "</think>" not in text:
        return text
    return text.rsplit("</think>", 1)[1]


def numeric_normalize(text: str) -> Optional[str]:
    normalized = normalize_candidate_text(text)
    if not normalized:
        return None
    match = re.fullmatch(r"[-+]?\d+", normalized)
    if match:
        return match.group(0).lstrip("+")
    return None


def candidate_solution_texts(solution_str: str) -> List[Tuple[str, str]]:
    final_tail = normalize_candidate_text(extract_tail_after_final_think(solution_str))
    full_text = normalize_candidate_text(solution_str)
    candidates: List[Tuple[str, str]] = []

    final_tail_boxed = extract_last_boxed_answer(final_tail) if final_tail else None
    if final_tail_boxed:
        candidates.append(("boxed_after_final_think", final_tail_boxed))
    if final_tail:
        candidates.append(("tail_after_final_think", final_tail))

    full_text_boxed = extract_last_boxed_answer(full_text) if full_text else None
    if full_text_boxed:
        candidates.append(("boxed_in_full_text", full_text_boxed))
    if full_text:
        candidates.append(("full_text", full_text))

    unique_candidates: List[Tuple[str, str]] = []
    seen: set[str] = set()
    for reason, candidate in candidates:
        normalized = normalize_candidate_text(candidate)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_candidates.append((reason, normalized))
    return unique_candidates


def is_max_length_finish_reason(finish_reason: Any) -> bool:
    if isinstance(finish_reason, str):
        return finish_reason.lower() in {"length", "max_length"}
    if isinstance(finish_reason, dict):
        reason_type = str(finish_reason.get("type", "")).lower()
        matched = str(finish_reason.get("matched", "")).lower()
        return reason_type in {"length", "max_length"} or matched in {"length", "max_length"}
    return False


def score_response(response_text: str, ground_truth: str) -> ScoreResult:
    from math_verify import parse, verify

    normalized_ground_truth = normalize_candidate_text(ground_truth)
    parsed_ground_truth = None
    try:
        parsed_ground_truth = parse(f"\\boxed{{{ground_truth}}}", parsing_timeout=30)
    except Exception as exc:
        ground_truth_parse_error = str(exc)
    else:
        ground_truth_parse_error = None

    attempts: List[Dict[str, Any]] = []
    for candidate_reason, candidate_text in candidate_solution_texts(response_text):
        candidate_record: Dict[str, Any] = {
            "candidate_reason": candidate_reason,
            "candidate_text": candidate_text,
        }

        candidate_integer = numeric_normalize(candidate_text)
        if candidate_integer is not None and candidate_integer == numeric_normalize(ground_truth):
            candidate_record["result"] = "exact_integer_match"
            return ScoreResult(
                score=1.0,
                reason="exact_integer_match",
                extracted_answer=candidate_text,
                debug={"attempts": attempts + [candidate_record]},
            )

        if normalize_candidate_text(candidate_text) == normalized_ground_truth:
            candidate_record["result"] = "exact_normalized_match"
            return ScoreResult(
                score=1.0,
                reason="exact_normalized_match",
                extracted_answer=candidate_text,
                debug={"attempts": attempts + [candidate_record]},
            )

        try:
            parsed_solution = parse(candidate_text, parsing_timeout=5)
        except Exception as exc:
            candidate_record["parse_error"] = str(exc)
            attempts.append(candidate_record)
            continue

        candidate_record["parsed_solution"] = repr(parsed_solution)
        if not isinstance(parsed_solution, (list, tuple)) or len(parsed_solution) < 2:
            candidate_record["result"] = "unusable_parse_result"
            attempts.append(candidate_record)
            continue

        if normalize_candidate_text(str(parsed_solution[1])) == normalized_ground_truth:
            candidate_record["result"] = "parsed_direct_match"
            return ScoreResult(
                score=1.0,
                reason="parsed_direct_match",
                extracted_answer=candidate_text,
                debug={"attempts": attempts + [candidate_record]},
            )

        if parsed_ground_truth is None:
            candidate_record["verify_error"] = (
                ground_truth_parse_error or "ground truth parse unavailable"
            )
            attempts.append(candidate_record)
            continue

        try:
            verified = bool(verify(parsed_ground_truth, parsed_solution, timeout_seconds=180))
        except Exception as exc:
            candidate_record["verify_error"] = str(exc)
            attempts.append(candidate_record)
            continue

        candidate_record["verified"] = verified
        if verified:
            candidate_record["result"] = "symbolic_verify_match"
            return ScoreResult(
                score=1.0,
                reason="symbolic_verify_match",
                extracted_answer=candidate_text,
                debug={"attempts": attempts + [candidate_record]},
            )

        candidate_record["result"] = "symbolic_verify_no_match"
        attempts.append(candidate_record)

    return ScoreResult(
        score=0.0,
        reason="no_valid_match",
        extracted_answer=attempts[0]["candidate_text"] if attempts else None,
        debug={
            "attempts": attempts,
            "ground_truth_parse_error": ground_truth_parse_error,
        },
    )


def base_sampling_params(
    max_new_tokens: int,
    *,
    enable_soft_thinking: bool = True,
) -> Dict[str, Any]:
    thinking_temperature = 0.8
    thinking_top_p = 0.95
    after_thinking_temperature = 0.6
    after_thinking_top_p = 0.95
    params = {
        "max_new_tokens": max_new_tokens,
        "temperature": thinking_temperature,
        "top_p": thinking_top_p,
        "top_k": -1,
        "min_p": 0.0,
    }
    if not enable_soft_thinking:
        return params
    params.update(
        {
            "after_thinking_temperature": after_thinking_temperature,
            "after_thinking_top_p": after_thinking_top_p,
            "after_thinking_top_k": -1,
            "after_thinking_min_p": 0.0,
            "think_end_str": "</think>",
            "max_topk": 3,
            "used_topk": 3,
            "enable_max_topk": False,
            "enable_gumbel": False,
            "enable_gumbel_after_thinking": False,
            "enable_replacement": True,
            "enable_unweighting": False,
            "enable_entropy_mask": False,
            "entropy_mask_threshold": 0.0,
            "early_stopping_entropy_threshold": 0.0,
            "early_stopping_length_threshold": 256,
        }
    )
    return params


def fixed_prefix_sampling_params(reasoning_prefix_tokens: int) -> Dict[str, Any]:
    params = base_sampling_params(
        max_new_tokens=reasoning_prefix_tokens,
        enable_soft_thinking=True,
    )
    params["think_end_str"] = FIXED_PREFIX_NEVER_SWITCH_THINK_END
    params.pop("stop", None)
    return params


def child_sampling_params(max_new_tokens: int) -> Dict[str, Any]:
    baseline_params = base_sampling_params(max_new_tokens=max_new_tokens, enable_soft_thinking=True)
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": baseline_params["after_thinking_temperature"],
        "top_p": baseline_params["after_thinking_top_p"],
        "top_k": baseline_params["after_thinking_top_k"],
        "min_p": baseline_params["after_thinking_min_p"],
        "custom_params": {"__disable_soft_thinking__": True},
    }


def standard_generation_sampling_params(max_new_tokens: int) -> Dict[str, Any]:
    # The plain-generation baseline uses the same non-multiplex decoding settings
    # that the fixed-prefix methods use after the multiplex reasoning prefix.
    return child_sampling_params(max_new_tokens)


def warmup_sampling_params() -> Dict[str, Any]:
    params = base_sampling_params(max_new_tokens=1)
    params["custom_params"] = {"__disable_soft_thinking__": True}
    return params


def verify_multiplex_runtime(
    *,
    client: SGLangRestClient,
    tokenizer: Any,
    args: argparse.Namespace,
    logger: logging.Logger,
    event_logger: StructuredEventLogger,
) -> MultiplexSelfCheckResult:
    server_info = client.get_server_info()
    enable_soft_thinking = server_info.get("enable_soft_thinking")
    attention_backend = server_info.get("attention_backend")

    if not enable_soft_thinking:
        message = "Server reports enable_soft_thinking=False; refusing to run a non-multiplex evaluation."
        event_logger.log(
            "multiplex_self_check_failed",
            reason="enable_soft_thinking_disabled",
            attention_backend=attention_backend,
            server_info=server_info,
        )
        return MultiplexSelfCheckResult(
            success=False,
            message=message,
            attention_backend=attention_backend,
            enable_soft_thinking=bool(enable_soft_thinking),
            has_topk_metadata=False,
            finish_reason=None,
            output_text="",
        )

    if attention_backend and attention_backend not in MULTIPLEX_REQUIRED_ATTENTION_BACKENDS:
        message = (
            f"Server resolved attention_backend={attention_backend!r}; this run requires a multiplex-capable backend "
            f"{sorted(MULTIPLEX_REQUIRED_ATTENTION_BACKENDS)!r}."
        )
        event_logger.log(
            "multiplex_self_check_failed",
            reason="non_multiplex_backend",
            attention_backend=attention_backend,
            server_info=server_info,
        )
        return MultiplexSelfCheckResult(
            success=False,
            message=message,
            attention_backend=attention_backend,
            enable_soft_thinking=bool(enable_soft_thinking),
            has_topk_metadata=False,
            finish_reason=None,
            output_text="",
        )

    prompt_text = build_prompt("What is 1+1?")
    prompt_ids, prompt_build_info = build_prefilled_prompt_ids(tokenizer, prompt_text)
    output = client.generate(
        input_ids=prompt_ids,
        sampling_params=base_sampling_params(max_new_tokens=8, enable_soft_thinking=True),
        rid="multiplex-self-check",
        data_parallel_rank=0,
    )
    if not isinstance(output, dict):
        message = f"Unexpected multiplex self-check response: {output!r}"
        event_logger.log("multiplex_self_check_failed", reason="unexpected_response", response=repr(output))
        return MultiplexSelfCheckResult(
            success=False,
            message=message,
            attention_backend=attention_backend,
            enable_soft_thinking=bool(enable_soft_thinking),
            has_topk_metadata=False,
            finish_reason=None,
            output_text="",
        )

    meta_info = output.get("meta_info", {})
    has_topk_metadata = (
        meta_info.get("output_topk_probs_list") is not None
        and meta_info.get("output_topk_indices_list") is not None
    )
    finish_reason = meta_info.get("finish_reason")
    output_text = reconstruct_assistant_text(
        prompt_build_info.assistant_prefill,
        output.get("text", ""),
    )
    if not has_topk_metadata:
        message = (
            "Soft-thinking self-check completed without top-k multiplex metadata; "
            "refusing to continue with a non-fidelity runtime."
        )
        event_logger.log(
            "multiplex_self_check_failed",
            reason="missing_topk_metadata",
            attention_backend=attention_backend,
            finish_reason=finish_reason,
            output_preview=output_text[:200],
        )
        return MultiplexSelfCheckResult(
            success=False,
            message=message,
            attention_backend=attention_backend,
            enable_soft_thinking=bool(enable_soft_thinking),
            has_topk_metadata=False,
            finish_reason=finish_reason,
            output_text=output_text,
        )

    logger.info(
        "[setup] multiplex self-check passed: attention_backend=%s finish_reason=%s",
        attention_backend or "auto",
        finish_reason,
    )
    event_logger.log(
        "multiplex_self_check_passed",
        attention_backend=attention_backend,
        finish_reason=finish_reason,
        prompt_build_mode=prompt_build_info.prompt_build_mode,
    )
    return MultiplexSelfCheckResult(
        success=True,
        message="ok",
        attention_backend=attention_backend,
        enable_soft_thinking=bool(enable_soft_thinking),
        has_topk_metadata=True,
        finish_reason=finish_reason,
        output_text=output_text,
    )


def chunk_ranges(total: int, chunk_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, total, chunk_size):
        yield start, min(start + chunk_size, total)


def normalize_generate_outputs(outputs: Any, expected_len: int, label: str) -> List[Dict[str, Any]]:
    if expected_len == 1 and isinstance(outputs, dict):
        return [outputs]
    if not isinstance(outputs, list) or len(outputs) != expected_len:
        raise RuntimeError(f"Unexpected {label} output: {outputs!r}")
    if not all(isinstance(output, dict) for output in outputs):
        raise RuntimeError(f"Unexpected {label} output payloads: {outputs!r}")
    return outputs


def make_target_dp_rank(prompt_index: int, dp_size: int) -> int:
    return prompt_index % max(dp_size, 1)


def planned_child_ids(parent_rid: str, max_k: int) -> List[str]:
    return [f"{parent_rid}-child-{index}" for index in range(max_k)]


def planned_child_seeds(seed: int, prompt_index: int, max_k: int) -> List[int]:
    base = seed * 100000 + prompt_index * 1000
    return [base + index for index in range(max_k)]


def make_sample_record(
    *,
    benchmark: str,
    method: str,
    example: Example,
    sample_index: int,
    rid: str,
    target_dp_rank: int,
    output: Dict[str, Any],
    planned_seed: Optional[int] = None,
    parent_trace: Optional[ParentTrace] = None,
    max_new_tokens: int,
    accepted_attempt_index: Optional[int] = None,
    prefix_text: Optional[str] = None,
    prefix_completion_tokens: int = 0,
) -> SampleRecord:
    meta_info = output.get("meta_info", {})
    generated_suffix = output.get("text", "")
    prefix = (
        prefix_text
        if prefix_text is not None
        else (parent_trace.shared_trace_text if parent_trace is not None else example.assistant_prefill)
    )
    full_text = reconstruct_assistant_text(prefix, generated_suffix)
    completion_tokens = int(meta_info.get("completion_tokens", 0))
    finish_reason = meta_info.get("finish_reason")
    usable_for_eval = True
    excluded_reason = None
    score_result = score_response(full_text, example.answer)
    return SampleRecord(
        benchmark=benchmark,
        method=method,
        prompt_index=example.prompt_index,
        problem_id=example.problem_id,
        sample_index=sample_index,
        rid=rid,
        target_dp_rank=target_dp_rank,
        correct=bool(score_result.score >= 1.0),
        score=score_result.score,
        text=full_text,
        finish_reason=finish_reason,
        prompt_tokens=int(meta_info.get("prompt_tokens", 0)),
        completion_tokens=completion_tokens,
        prefix_completion_tokens=prefix_completion_tokens,
        cached_tokens=int(meta_info.get("cached_tokens", 0)),
        latency_seconds=float(meta_info.get("e2e_latency", 0.0)),
        session_id=parent_trace.session_id if parent_trace else None,
        parent_rid=parent_trace.parent_rid if parent_trace else None,
        parent_completion_tokens=parent_trace.completion_tokens if parent_trace else 0,
        cacheable_token_count=parent_trace.cacheable_token_count if parent_trace else 0,
        cache_verification_passed=(
            int(meta_info.get("cached_tokens", 0)) >= parent_trace.cacheable_token_count
            if parent_trace is not None
            else None
        ),
        planned_seed=planned_seed,
        usable_for_eval=usable_for_eval,
        excluded_reason=excluded_reason,
        score_reason=score_result.reason,
        extracted_answer=score_result.extracted_answer,
        score_debug=score_result.debug,
        generated_suffix=generated_suffix,
        accepted_attempt_index=accepted_attempt_index,
        reasoning_prefix_tokens=(
            parent_trace.reasoning_prefix_tokens if parent_trace is not None else prefix_completion_tokens
        ),
    )


def determine_eot_retry_reject_reason(
    full_text: str,
    finish_reason: Any,
    completion_tokens: int,
    max_new_tokens: int,
) -> Optional[str]:
    if THINK_END_TAG not in full_text:
        return "missing_think_end"
    if is_max_length_finish_reason(finish_reason) or completion_tokens >= max_new_tokens:
        return "max_new_tokens_reached"
    return None


def reached_reasoning_prefix(output: Dict[str, Any], reasoning_prefix_tokens: int) -> bool:
    completion_tokens = int(output.get("meta_info", {}).get("completion_tokens", 0))
    return completion_tokens >= reasoning_prefix_tokens


def continuation_token_budget(max_new_tokens: int, reasoning_prefix_tokens: int) -> int:
    remaining = max_new_tokens - reasoning_prefix_tokens
    if remaining < 1:
        raise ValueError(
            f"max_new_tokens={max_new_tokens} must exceed reasoning_prefix_tokens={reasoning_prefix_tokens}."
        )
    return remaining


def planned_attempt_seed(
    seed: int,
    prompt_index: int,
    slot_index: int,
    attempt_index: int,
) -> int:
    return (seed * 1_000_000) + (prompt_index * 10_000) + (slot_index * 100) + attempt_index


def make_attempt_record(
    *,
    benchmark: str,
    method: str,
    target_kind: str,
    example: Example,
    slot_index: Optional[int],
    attempt_index: int,
    rid: str,
    target_dp_rank: int,
    output: Dict[str, Any],
    prefix_text: str,
    max_new_tokens: int,
    planned_seed: Optional[int],
    cumulative_attempts: int,
    cumulative_completion_tokens: int,
    budget_exhausted: bool,
    terminal: bool,
) -> AttemptRecord:
    meta_info = output.get("meta_info", {})
    generated_suffix = output.get("text", "")
    full_text = reconstruct_assistant_text(prefix_text, generated_suffix)
    completion_tokens = int(meta_info.get("completion_tokens", 0))
    finish_reason = meta_info.get("finish_reason")
    reject_reason = determine_eot_retry_reject_reason(
        full_text,
        finish_reason,
        completion_tokens,
        max_new_tokens,
    )
    return AttemptRecord(
        benchmark=benchmark,
        method=method,
        prompt_index=example.prompt_index,
        problem_id=example.problem_id,
        target_kind=target_kind,
        slot_index=slot_index,
        attempt_index=attempt_index,
        rid=rid,
        target_dp_rank=target_dp_rank,
        accepted_for_eval=reject_reason is None,
        reject_reason=reject_reason,
        finish_reason=finish_reason,
        prompt_tokens=int(meta_info.get("prompt_tokens", 0)),
        completion_tokens=completion_tokens,
        cached_tokens=int(meta_info.get("cached_tokens", 0)),
        latency_seconds=float(meta_info.get("e2e_latency", 0.0)),
        planned_seed=planned_seed,
        contains_think_end=THINK_END_TAG in full_text,
        cumulative_attempts=cumulative_attempts,
        cumulative_completion_tokens=cumulative_completion_tokens,
        budget_exhausted=budget_exhausted,
        terminal=terminal,
        generated_suffix=generated_suffix,
    )


def reject_reason_counts(records: Iterable[AttemptRecord]) -> Dict[str, int]:
    counter: Counter[str] = Counter()
    for record in records:
        if record.reject_reason:
            counter[record.reject_reason] += 1
    return dict(counter)


def ensure_output_dir(args: argparse.Namespace) -> Path:
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def append_records(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    writer = JsonlWriter(path)
    for record in records:
        writer.append(record)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def determine_completed_prompts(
    baseline_prompt_rows: List[Dict[str, Any]],
    parent_rows: List[Dict[str, Any]],
) -> Dict[Tuple[str, int], bool]:
    completed: Dict[Tuple[str, int], bool] = {}
    for row in baseline_prompt_rows:
        method = str(row.get("method", "baseline_independent"))
        completed[(method, int(row["prompt_index"]))] = True
    for row in parent_rows:
        completed[("shared_trace_branch_after_prefix", int(row["prompt_index"]))] = True
    return completed


def sample_row_usable_for_eval(row: Dict[str, Any]) -> bool:
    if "usable_for_eval" in row:
        return bool(row["usable_for_eval"])
    if is_max_length_finish_reason(row.get("finish_reason")):
        return False
    return True


def sample_row_excluded_reason(row: Dict[str, Any]) -> Optional[str]:
    if "excluded_reason" in row:
        return row.get("excluded_reason")
    if is_max_length_finish_reason(row.get("finish_reason")):
        return "max_new_tokens_reached"
    return None


def run_baseline_for_prompt(
    *,
    args: argparse.Namespace,
    benchmark: str,
    client: SGLangRestClient,
    example: Example,
    target_dp_rank: int,
    logger: logging.Logger,
    event_logger: StructuredEventLogger,
) -> Tuple[BaselinePromptResult, List[SampleRecord], List[AttemptRecord]]:
    reasoning_prefix_tokens = int(args.current_reasoning_prefix_tokens)
    decode_budget = continuation_token_budget(args.max_new_tokens, reasoning_prefix_tokens)
    logger.info(
        "[baseline-fixed-prefix] Prompt %s: generating %s independent traces with %s multiplex prefix tokens on DP rank %s, then switching to discrete decoding for %s tokens.",
        example.prompt_index,
        args.max_k,
        reasoning_prefix_tokens,
        target_dp_rank,
        decode_budget,
    )
    event_logger.log(
        "baseline_prompt_start",
        prompt_index=example.prompt_index,
        target_dp_rank=target_dp_rank,
        required_sample_count=args.max_k,
        reasoning_prefix_tokens=reasoning_prefix_tokens,
        decode_max_new_tokens=decode_budget,
    )
    prefix_params = fixed_prefix_sampling_params(reasoning_prefix_tokens)
    continuation_params = child_sampling_params(decode_budget)
    continuation_seeds = planned_child_seeds(args.seed, example.prompt_index, args.max_k)
    slot_statuses: List[Dict[str, Any]] = [
        {
            "slot_index": slot_index,
            "prefix_completion_tokens": 0,
            "prefix_latency_seconds": 0.0,
            "prefix_finish_reason": None,
            "prefix_reached": False,
            "continuation_completion_tokens": 0,
            "continuation_latency_seconds": 0.0,
            "accepted_for_eval": False,
            "failure_reason": None,
        }
        for slot_index in range(args.max_k)
    ]
    accepted_records: List[SampleRecord] = []
    session_ids = [
        f"baseline-p{example.prompt_index}-t{reasoning_prefix_tokens}-slot{slot_index}"
        for slot_index in range(args.max_k)
    ]
    prefix_rids = [f"{session_id}-prefix" for session_id in session_ids]
    continuation_rids = [f"{session_id}-continuation" for session_id in session_ids]
    opened_session_ids: List[str] = []
    continuation_jobs: List[Dict[str, Any]] = []
    try:
        for session_id in session_ids:
            client.open_session(args.capacity_of_str_len, session_id=session_id)
            opened_session_ids.append(session_id)
        event_logger.log(
            "baseline_prefix_batch_start",
            prompt_index=example.prompt_index,
            target_dp_rank=target_dp_rank,
            reasoning_prefix_tokens=reasoning_prefix_tokens,
            batch_size=args.max_k,
        )
        prefix_outputs = normalize_generate_outputs(
            client.generate(
                input_ids=[example.prompt_ids] * args.max_k,
                sampling_params=[dict(prefix_params) for _ in range(args.max_k)],
                rid=prefix_rids,
                data_parallel_rank=target_dp_rank,
                session_params=[{"id": session_id} for session_id in session_ids],
            ),
            args.max_k,
            "baseline prefix batch",
        )
        event_logger.log(
            "baseline_prefix_batch_done",
            prompt_index=example.prompt_index,
            target_dp_rank=target_dp_rank,
            reasoning_prefix_tokens=reasoning_prefix_tokens,
            batch_size=args.max_k,
        )

        for slot_index, prefix_output in enumerate(prefix_outputs):
            slot_status = slot_statuses[slot_index]
            prefix_meta = prefix_output.get("meta_info", {})
            prefix_completion_tokens = int(prefix_meta.get("completion_tokens", 0))
            prefix_latency_seconds = float(prefix_meta.get("e2e_latency", 0.0))
            prefix_finish_reason = prefix_meta.get("finish_reason")
            prefix_text = reconstruct_assistant_text(
                example.assistant_prefill,
                prefix_output.get("text", ""),
            )
            prefix_reached = reached_reasoning_prefix(prefix_output, reasoning_prefix_tokens)
            slot_status["prefix_completion_tokens"] = prefix_completion_tokens
            slot_status["prefix_latency_seconds"] = prefix_latency_seconds
            slot_status["prefix_finish_reason"] = prefix_finish_reason
            slot_status["prefix_reached"] = prefix_reached
            event_logger.log(
                "baseline_prefix_sample",
                prompt_index=example.prompt_index,
                slot_index=slot_index,
                rid=prefix_rids[slot_index],
                target_dp_rank=target_dp_rank,
                reasoning_prefix_tokens=reasoning_prefix_tokens,
                prefix_reached=prefix_reached,
                completion_tokens=prefix_completion_tokens,
                latency_seconds=prefix_latency_seconds,
                finish_reason=prefix_finish_reason,
            )
            if not prefix_reached:
                slot_status["failure_reason"] = "prefix_not_reached"
                continue

            fork_info = client.fork_request(
                session_id=session_ids[slot_index],
                parent_rid=prefix_rids[slot_index],
                child_count=1,
                child_rids=[continuation_rids[slot_index]],
                child_seeds=[continuation_seeds[slot_index]],
                target_dp_rank=target_dp_rank,
                allow_non_eot_branch=True,
            )
            if not fork_info.get("success", False):
                slot_status["failure_reason"] = "fork_failed"
                event_logger.log(
                    "baseline_prefix_fork_failed",
                    prompt_index=example.prompt_index,
                    slot_index=slot_index,
                    rid=prefix_rids[slot_index],
                    target_dp_rank=target_dp_rank,
                    reasoning_prefix_tokens=reasoning_prefix_tokens,
                    message=fork_info.get("message", "fork_request failed"),
                )
                continue
            continuation_jobs.append(
                {
                    "slot_index": slot_index,
                    "input_ids": fork_info["branch_input_ids"],
                    "rid": continuation_rids[slot_index],
                    "planned_seed": continuation_seeds[slot_index],
                    "prefix_text": prefix_text,
                    "prefix_completion_tokens": prefix_completion_tokens,
                }
            )

        for start, end in chunk_ranges(len(continuation_jobs), args.effective_request_batch_size):
            batch_jobs = continuation_jobs[start:end]
            event_logger.log(
                "baseline_continuation_batch_start",
                prompt_index=example.prompt_index,
                start_index=start,
                end_index=end,
                target_dp_rank=target_dp_rank,
                reasoning_prefix_tokens=reasoning_prefix_tokens,
            )
            continuation_outputs = normalize_generate_outputs(
                client.generate(
                    input_ids=[job["input_ids"] for job in batch_jobs],
                    sampling_params=[dict(continuation_params) for _ in batch_jobs],
                    rid=[job["rid"] for job in batch_jobs],
                    data_parallel_rank=target_dp_rank,
                ),
                len(batch_jobs),
                "baseline continuation batch",
            )
            event_logger.log(
                "baseline_continuation_batch_done",
                prompt_index=example.prompt_index,
                start_index=start,
                end_index=end,
                target_dp_rank=target_dp_rank,
                reasoning_prefix_tokens=reasoning_prefix_tokens,
            )
            for job, continuation_output in zip(batch_jobs, continuation_outputs):
                slot_index = int(job["slot_index"])
                slot_status = slot_statuses[slot_index]
                continuation_meta = continuation_output.get("meta_info", {})
                slot_status["continuation_completion_tokens"] = int(
                    continuation_meta.get("completion_tokens", 0)
                )
                slot_status["continuation_latency_seconds"] = float(
                    continuation_meta.get("e2e_latency", 0.0)
                )
                slot_status["accepted_for_eval"] = True
                record = make_sample_record(
                    benchmark=benchmark,
                    method="baseline_independent",
                    example=example,
                    sample_index=slot_index,
                    rid=str(job["rid"]),
                    target_dp_rank=target_dp_rank,
                    output=continuation_output,
                    planned_seed=int(job["planned_seed"]),
                    max_new_tokens=decode_budget,
                    accepted_attempt_index=None,
                    prefix_text=str(job["prefix_text"]),
                    prefix_completion_tokens=int(job["prefix_completion_tokens"]),
                )
                accepted_records.append(record)
                event_logger.log(
                    "baseline_continuation_sample",
                    prompt_index=example.prompt_index,
                    slot_index=slot_index,
                    rid=record.rid,
                    target_dp_rank=target_dp_rank,
                    reasoning_prefix_tokens=reasoning_prefix_tokens,
                    correct=record.correct,
                    score=record.score,
                    prefix_completion_tokens=record.prefix_completion_tokens,
                    completion_tokens=record.completion_tokens,
                    latency_seconds=record.latency_seconds,
                    finish_reason=record.finish_reason,
                )
    finally:
        for session_id in reversed(opened_session_ids):
            client.close_session(session_id)

    accepted_records.sort(key=lambda record: int(record.sample_index))
    success = len(accepted_records) == args.max_k
    missing_slots = [
        int(status["slot_index"]) for status in slot_statuses if not status["accepted_for_eval"]
    ]
    if success:
        message = (
            f"All baseline slots reached the fixed reasoning prefix ({reasoning_prefix_tokens} tokens) "
            "and completed discrete decoding."
        )
    else:
        message = (
            f"Fixed reasoning prefix was not reached for baseline slots "
            + ", ".join(str(slot_index) for slot_index in missing_slots)
        )
    reject_reason_counts_payload = dict(
        Counter(
            str(status["failure_reason"])
            for status in slot_statuses
            if status["failure_reason"]
        )
    )
    baseline_result = BaselinePromptResult(
        benchmark=benchmark,
        prompt_index=example.prompt_index,
        problem_id=example.problem_id,
        target_dp_rank=target_dp_rank,
        success=success,
        message=message,
        required_sample_count=args.max_k,
        usable_sample_count=len(accepted_records),
        attempts_used=args.max_k,
        total_completion_tokens_spent=sum(
            int(status["prefix_completion_tokens"]) + int(status["continuation_completion_tokens"])
            for status in slot_statuses
        ),
        total_latency_seconds_spent=sum(
            float(status["prefix_latency_seconds"]) + float(status["continuation_latency_seconds"])
            for status in slot_statuses
        ),
        reject_reason_counts=reject_reason_counts_payload,
        slot_statuses=slot_statuses,
        reasoning_prefix_tokens=reasoning_prefix_tokens,
        method="baseline_independent",
    )
    event_logger.log(
        "baseline_prompt_done",
        prompt_index=example.prompt_index,
        target_dp_rank=target_dp_rank,
        success=baseline_result.success,
        usable_sample_count=baseline_result.usable_sample_count,
        reasoning_prefix_tokens=reasoning_prefix_tokens,
        total_completion_tokens_spent=baseline_result.total_completion_tokens_spent,
        reject_reason_counts=baseline_result.reject_reason_counts,
    )
    return baseline_result, accepted_records, []


def run_standard_generation_for_prompt(
    *,
    args: argparse.Namespace,
    benchmark: str,
    client: SGLangRestClient,
    example: Example,
    target_dp_rank: int,
    logger: logging.Logger,
    event_logger: StructuredEventLogger,
) -> Tuple[BaselinePromptResult, List[SampleRecord], List[AttemptRecord]]:
    logger.info(
        "[standard-generation] Prompt %s: generating %s independent non-multiplex traces on DP rank %s.",
        example.prompt_index,
        args.max_k,
        target_dp_rank,
    )
    event_logger.log(
        "standard_generation_prompt_start",
        prompt_index=example.prompt_index,
        target_dp_rank=target_dp_rank,
        required_sample_count=args.max_k,
    )
    params = standard_generation_sampling_params(args.max_new_tokens)
    accepted_records: List[SampleRecord] = []
    slot_statuses: List[Dict[str, Any]] = [
        {
            "slot_index": slot_index,
            "prefix_completion_tokens": 0,
            "prefix_latency_seconds": 0.0,
            "prefix_finish_reason": None,
            "prefix_reached": True,
            "continuation_completion_tokens": 0,
            "continuation_latency_seconds": 0.0,
            "accepted_for_eval": False,
            "failure_reason": None,
        }
        for slot_index in range(args.max_k)
    ]

    for start, end in chunk_ranges(args.max_k, args.effective_request_batch_size):
        batch_size = end - start
        batch_rids = [
            f"standard-generation-p{example.prompt_index}-sample{sample_index}"
            for sample_index in range(start, end)
        ]
        event_logger.log(
            "standard_generation_batch_start",
            prompt_index=example.prompt_index,
            start_index=start,
            end_index=end,
            target_dp_rank=target_dp_rank,
        )
        outputs = normalize_generate_outputs(
            client.generate(
                input_ids=[example.prompt_ids] * batch_size,
                sampling_params=[dict(params) for _ in range(batch_size)],
                rid=batch_rids,
                data_parallel_rank=target_dp_rank,
            ),
            batch_size,
            "standard generation batch",
        )
        event_logger.log(
            "standard_generation_batch_done",
            prompt_index=example.prompt_index,
            start_index=start,
            end_index=end,
            target_dp_rank=target_dp_rank,
        )
        for offset, output in enumerate(outputs):
            sample_index = start + offset
            slot_status = slot_statuses[sample_index]
            meta_info = output.get("meta_info", {})
            slot_status["continuation_completion_tokens"] = int(
                meta_info.get("completion_tokens", 0)
            )
            slot_status["continuation_latency_seconds"] = float(
                meta_info.get("e2e_latency", 0.0)
            )
            slot_status["accepted_for_eval"] = True
            record = make_sample_record(
                benchmark=benchmark,
                method="standard_generation_independent",
                example=example,
                sample_index=sample_index,
                rid=batch_rids[offset],
                target_dp_rank=target_dp_rank,
                output=output,
                planned_seed=None,
                max_new_tokens=args.max_new_tokens,
                accepted_attempt_index=None,
                prefix_text=example.assistant_prefill,
                prefix_completion_tokens=0,
            )
            accepted_records.append(record)
            event_logger.log(
                "standard_generation_sample",
                prompt_index=example.prompt_index,
                sample_index=sample_index,
                rid=record.rid,
                target_dp_rank=target_dp_rank,
                correct=record.correct,
                score=record.score,
                completion_tokens=record.completion_tokens,
                latency_seconds=record.latency_seconds,
                finish_reason=record.finish_reason,
            )

    accepted_records.sort(key=lambda record: int(record.sample_index))
    prompt_result = BaselinePromptResult(
        benchmark=benchmark,
        prompt_index=example.prompt_index,
        problem_id=example.problem_id,
        target_dp_rank=target_dp_rank,
        success=len(accepted_records) == args.max_k,
        message="Completed standard non-multiplex generation.",
        required_sample_count=args.max_k,
        usable_sample_count=len(accepted_records),
        attempts_used=args.max_k,
        total_completion_tokens_spent=sum(
            int(status["continuation_completion_tokens"]) for status in slot_statuses
        ),
        total_latency_seconds_spent=sum(
            float(status["continuation_latency_seconds"]) for status in slot_statuses
        ),
        reject_reason_counts={},
        slot_statuses=slot_statuses,
        reasoning_prefix_tokens=0,
        method="standard_generation_independent",
    )
    event_logger.log(
        "standard_generation_prompt_done",
        prompt_index=example.prompt_index,
        target_dp_rank=target_dp_rank,
        usable_sample_count=prompt_result.usable_sample_count,
        total_completion_tokens_spent=prompt_result.total_completion_tokens_spent,
    )
    return prompt_result, accepted_records, []


def run_shared_trace_for_prompt(
    *,
    args: argparse.Namespace,
    benchmark: str,
    client: SGLangRestClient,
    tokenizer: Any,
    example: Example,
    target_dp_rank: int,
    logger: logging.Logger,
    event_logger: StructuredEventLogger,
) -> Tuple[ParentTrace, List[SampleRecord], List[AttemptRecord]]:
    reasoning_prefix_tokens = int(args.current_reasoning_prefix_tokens)
    decode_budget = continuation_token_budget(args.max_new_tokens, reasoning_prefix_tokens)
    logger.info(
        "[shared-trace-fixed-prefix] Prompt %s: generating one shared multiplex prefix of %s tokens on DP rank %s, then branching into %s discrete continuations of %s tokens.",
        example.prompt_index,
        reasoning_prefix_tokens,
        target_dp_rank,
        args.max_k,
        decode_budget,
    )
    parent_params = fixed_prefix_sampling_params(reasoning_prefix_tokens)
    session_id = f"aime-shared-p{example.prompt_index}-t{reasoning_prefix_tokens}"
    parent_rid = f"shared-prefix-p{example.prompt_index}-t{reasoning_prefix_tokens}"
    session_opened = False
    parent_output: Optional[Dict[str, Any]] = None
    parent_text = ""
    parent_generated_suffix = ""
    finish_reason: Any = None
    try:
        client.open_session(args.capacity_of_str_len, session_id=session_id)
        session_opened = True
        event_logger.log(
            "shared_parent_start",
            prompt_index=example.prompt_index,
            target_dp_rank=target_dp_rank,
            session_id=session_id,
            parent_rid=parent_rid,
            reasoning_prefix_tokens=reasoning_prefix_tokens,
            decode_max_new_tokens=decode_budget,
        )
        parent_output = client.generate(
            input_ids=example.prompt_ids,
            sampling_params=parent_params,
            rid=parent_rid,
            data_parallel_rank=target_dp_rank,
            session_params={"id": session_id},
        )
        if not isinstance(parent_output, dict):
            raise RuntimeError(f"Unexpected shared prefix output: {parent_output!r}")
        parent_generated_suffix = parent_output.get("text", "")
        parent_text = reconstruct_assistant_text(
            example.assistant_prefill,
            parent_generated_suffix,
        )
        parent_meta = parent_output.get("meta_info", {})
        finish_reason = parent_meta.get("finish_reason")
        parent_completion_tokens = int(parent_meta.get("completion_tokens", 0))
        parent_latency_seconds = float(parent_meta.get("e2e_latency", 0.0))
        prefix_reached = reached_reasoning_prefix(parent_output, reasoning_prefix_tokens)
        event_logger.log(
            "shared_parent_done",
            prompt_index=example.prompt_index,
            target_dp_rank=target_dp_rank,
            session_id=session_id,
            parent_rid=parent_rid,
            reasoning_prefix_tokens=reasoning_prefix_tokens,
            prefix_reached=prefix_reached,
            completion_tokens=parent_completion_tokens,
            latency_seconds=parent_latency_seconds,
            finish_reason=finish_reason,
        )
        if not prefix_reached:
            message = (
                f"Shared prefix stopped before reaching the fixed reasoning prefix of "
                f"{reasoning_prefix_tokens} tokens."
            )
            logger.warning("[shared-trace-fixed-prefix] Prompt %s: %s", example.prompt_index, message)
            return (
                ParentTrace(
                    benchmark=benchmark,
                    prompt_index=example.prompt_index,
                    problem_id=example.problem_id,
                    session_id=session_id,
                    parent_rid=parent_rid,
                    target_dp_rank=target_dp_rank,
                    success=False,
                    message=message,
                    shared_trace_text=parent_text,
                    branch_input_ids=[],
                    cacheable_input_ids=[],
                    uncached_tail_input_ids=[],
                    cacheable_token_count=0,
                    eot_token_id=None,
                    eot_output_index=-1,
                    prompt_token_count=int(parent_meta.get("prompt_tokens", 0)),
                    response_token_count=parent_completion_tokens,
                    completion_tokens=parent_completion_tokens,
                    finish_reason=finish_reason,
                    verification={
                        "requested_reasoning_prefix_tokens": reasoning_prefix_tokens,
                        "prefix_reached": False,
                    },
                    usable_for_eval=False,
                    excluded_reason="prefix_not_reached",
                    generated_suffix=parent_generated_suffix,
                    attempts_used=1,
                    total_completion_tokens_spent=parent_completion_tokens,
                    total_latency_seconds_spent=parent_latency_seconds,
                    reject_reason_counts={"prefix_not_reached": 1},
                    accepted_attempt_index=None,
                    reasoning_prefix_tokens=reasoning_prefix_tokens,
                ),
                [],
                [],
            )

        child_ids = planned_child_ids(parent_rid, args.max_k)
        child_seeds = planned_child_seeds(args.seed, example.prompt_index, args.max_k)
        fork_info = client.fork_request(
            session_id=session_id,
            parent_rid=parent_rid,
            child_count=args.max_k,
            child_rids=child_ids,
            child_seeds=child_seeds,
            target_dp_rank=target_dp_rank,
            allow_non_eot_branch=True,
        )
        event_logger.log(
            "kv_fork_prefix",
            prompt_index=example.prompt_index,
            session_id=session_id,
            parent_rid=parent_rid,
            target_dp_rank=target_dp_rank,
            reasoning_prefix_tokens=reasoning_prefix_tokens,
            child_count=args.max_k,
            cacheable_token_count=fork_info.get("cacheable_token_count", 0),
            uncached_tail_len=len(fork_info.get("uncached_tail_input_ids", [])),
            message=fork_info.get("message"),
        )
        if not fork_info.get("success", False):
            message = fork_info.get("message", "fork_request failed")
            return (
                ParentTrace(
                    benchmark=benchmark,
                    prompt_index=example.prompt_index,
                    problem_id=example.problem_id,
                    session_id=session_id,
                    parent_rid=parent_rid,
                    target_dp_rank=target_dp_rank,
                    success=False,
                    message=message,
                    shared_trace_text=parent_text,
                    branch_input_ids=[],
                    cacheable_input_ids=[],
                    uncached_tail_input_ids=[],
                    cacheable_token_count=0,
                    eot_token_id=None,
                    eot_output_index=-1,
                    prompt_token_count=int(parent_meta.get("prompt_tokens", 0)),
                    response_token_count=parent_completion_tokens,
                    completion_tokens=parent_completion_tokens,
                    finish_reason=finish_reason,
                    verification={
                        "requested_reasoning_prefix_tokens": reasoning_prefix_tokens,
                        "prefix_reached": True,
                    },
                    usable_for_eval=False,
                    excluded_reason="fork_failed",
                    generated_suffix=parent_generated_suffix,
                    attempts_used=1,
                    total_completion_tokens_spent=parent_completion_tokens,
                    total_latency_seconds_spent=parent_latency_seconds,
                    reject_reason_counts={"fork_failed": 1},
                    accepted_attempt_index=None,
                    reasoning_prefix_tokens=reasoning_prefix_tokens,
                ),
                [],
                [],
            )

        parent_trace = ParentTrace(
            benchmark=benchmark,
            prompt_index=example.prompt_index,
            problem_id=example.problem_id,
            session_id=session_id,
            parent_rid=parent_rid,
            target_dp_rank=target_dp_rank,
            success=True,
            message=fork_info.get("message", ""),
            shared_trace_text=parent_text,
            branch_input_ids=fork_info["branch_input_ids"],
            cacheable_input_ids=fork_info["cacheable_input_ids"],
            uncached_tail_input_ids=fork_info["uncached_tail_input_ids"],
            cacheable_token_count=int(fork_info.get("cacheable_token_count", 0)),
            eot_token_id=None,
            eot_output_index=-1,
            prompt_token_count=int(parent_meta.get("prompt_tokens", 0)),
            response_token_count=parent_completion_tokens,
            completion_tokens=parent_completion_tokens,
            finish_reason=finish_reason,
            verification={
                "requested_reasoning_prefix_tokens": reasoning_prefix_tokens,
                "prefix_reached": True,
            },
            usable_for_eval=True,
            excluded_reason=None,
            generated_suffix=parent_generated_suffix,
            attempts_used=1,
            total_completion_tokens_spent=parent_completion_tokens,
            total_latency_seconds_spent=parent_latency_seconds,
            reject_reason_counts={},
            accepted_attempt_index=None,
            reasoning_prefix_tokens=reasoning_prefix_tokens,
        )

        child_records: List[SampleRecord] = []
        params = child_sampling_params(decode_budget)
        for start, end in chunk_ranges(args.max_k, args.effective_request_batch_size):
            batch_len = end - start
            batch_rids = child_ids[start:end]
            event_logger.log(
                "shared_children_batch_start",
                prompt_index=example.prompt_index,
                start_index=start,
                end_index=end,
                target_dp_rank=target_dp_rank,
                session_id=session_id,
                parent_rid=parent_rid,
                reasoning_prefix_tokens=reasoning_prefix_tokens,
            )
            outputs = client.generate(
                input_ids=[fork_info["branch_input_ids"]] * batch_len,
                sampling_params=[params] * batch_len,
                rid=batch_rids,
                data_parallel_rank=target_dp_rank,
            )
            if not isinstance(outputs, list) or len(outputs) != batch_len:
                raise RuntimeError(f"Unexpected child batch output: {outputs!r}")
            batch_records: List[SampleRecord] = []
            for offset, output in enumerate(outputs):
                sample_index = start + offset
                record = make_sample_record(
                    benchmark=benchmark,
                    method="shared_trace_branch_after_prefix",
                    example=example,
                    sample_index=sample_index,
                    rid=batch_rids[offset],
                    target_dp_rank=target_dp_rank,
                    output=output,
                    planned_seed=child_seeds[sample_index],
                    parent_trace=parent_trace,
                    max_new_tokens=decode_budget,
                    accepted_attempt_index=None,
                )
                batch_records.append(record)
                child_records.append(record)
                event_logger.log(
                    "shared_child_sample",
                    prompt_index=example.prompt_index,
                    sample_index=sample_index,
                    rid=record.rid,
                    session_id=record.session_id,
                    parent_rid=record.parent_rid,
                    target_dp_rank=record.target_dp_rank,
                    planned_seed=record.planned_seed,
                    reasoning_prefix_tokens=reasoning_prefix_tokens,
                    correct=record.correct,
                    score=record.score,
                    prompt_tokens=record.prompt_tokens,
                    completion_tokens=record.completion_tokens,
                    cached_tokens=record.cached_tokens,
                    latency_seconds=record.latency_seconds,
                    cacheable_token_count=record.cacheable_token_count,
                    cache_verification_passed=record.cache_verification_passed,
                    finish_reason=record.finish_reason,
                )
            event_logger.log(
                "shared_children_batch_done",
                prompt_index=example.prompt_index,
                start_index=start,
                end_index=end,
                target_dp_rank=target_dp_rank,
                reasoning_prefix_tokens=reasoning_prefix_tokens,
                cacheable_token_count=parent_trace.cacheable_token_count,
                average_cached_tokens=sum(record.cached_tokens for record in batch_records) / batch_len,
                all_cache_checks_passed=all(bool(record.cache_verification_passed) for record in batch_records),
            )
        return parent_trace, child_records, []
    finally:
        if session_opened:
            client.close_session(session_id)
            event_logger.log(
                "shared_session_closed",
                prompt_index=example.prompt_index,
                session_id=session_id,
                parent_rid=parent_rid,
                parent_finish_reason=(
                    parent_output.get("meta_info", {}).get("finish_reason")
                    if isinstance(parent_output, dict)
                    else None
                ),
            )


def compute_summary_tables(
    *,
    sample_rows: List[Dict[str, Any]],
    baseline_prompt_rows: List[Dict[str, Any]],
    parent_rows: List[Dict[str, Any]],
    attempt_rows: List[Dict[str, Any]],
    pass_at_ks: List[int],
    max_k: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    grouped_samples: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in sample_rows:
        key = (row["method"], int(row["prompt_index"]))
        grouped_samples.setdefault(key, []).append(row)

    grouped_attempts: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in attempt_rows:
        key = (row["method"], int(row["prompt_index"]))
        grouped_attempts.setdefault(key, []).append(row)

    prompt_result_by_key = {
        (str(row.get("method", "baseline_independent")), int(row["prompt_index"])): row
        for row in baseline_prompt_rows
    }
    parent_by_prompt = {
        int(row["prompt_index"]): row for row in parent_rows
    }

    all_method_prompt_rows: List[Dict[str, Any]] = []
    prompt_indices = sorted(
        {
            int(row["prompt_index"]) for row in sample_rows
        }
        | {int(row["prompt_index"]) for row in baseline_prompt_rows}
        | set(parent_by_prompt.keys())
        | {int(row["prompt_index"]) for row in attempt_rows}
    )
    methods = sorted(
        {str(row["method"]) for row in sample_rows}
        | {str(row.get("method", "baseline_independent")) for row in baseline_prompt_rows}
        | (
            {"shared_trace_branch_after_prefix"}
            if parent_rows
            else set()
        )
    )

    for method in methods:
        for prompt_index in prompt_indices:
            samples = sorted(
                grouped_samples.get((method, prompt_index), []),
                key=lambda row: int(row["sample_index"]),
            )
            usable_samples = [row for row in samples if sample_row_usable_for_eval(row)]
            excluded_reasons = [
                reason
                for reason in (sample_row_excluded_reason(row) for row in samples)
                if reason
            ]
            attempt_rows_for_prompt = grouped_attempts.get((method, prompt_index), [])
            parent = parent_by_prompt.get(prompt_index)
            prompt_result = prompt_result_by_key.get((method, prompt_index))
            parent_completion_tokens = (
                int(parent["completion_tokens"])
                if method == "shared_trace_branch_after_prefix" and parent
                else 0
            )
            success = True
            error = ""
            attempts_used = len(attempt_rows_for_prompt)
            total_attempt_completion_tokens = sum(
                int(row.get("completion_tokens", 0)) for row in attempt_rows_for_prompt
            )
            total_attempt_latency_seconds = sum(
                float(row.get("latency_seconds", 0.0)) for row in attempt_rows_for_prompt
            )
            if prompt_result is not None:
                success = bool(prompt_result.get("success", False))
                error = str(prompt_result.get("message", ""))
                attempts_used = int(prompt_result.get("attempts_used", attempts_used))
                total_attempt_completion_tokens = int(
                    prompt_result.get(
                        "total_completion_tokens_spent",
                        total_attempt_completion_tokens,
                    )
                )
                total_attempt_latency_seconds = float(
                    prompt_result.get(
                        "total_latency_seconds_spent",
                        total_attempt_latency_seconds,
                    )
                )
                for reason in (prompt_result.get("reject_reason_counts") or {}).keys():
                    excluded_reasons.append(str(reason))
            elif method == "shared_trace_branch_after_prefix" and parent is not None:
                success = bool(parent.get("success", False))
                error = str(parent.get("message", ""))
                attempts_used = int(parent.get("attempts_used", attempts_used))
                total_attempt_completion_tokens = int(
                    parent.get("total_completion_tokens_spent", total_attempt_completion_tokens)
                )
                total_attempt_latency_seconds = float(
                    parent.get("total_latency_seconds_spent", total_attempt_latency_seconds)
                )
                if parent.get("excluded_reason"):
                    excluded_reasons.append(str(parent["excluded_reason"]))
                for reason in (parent.get("reject_reason_counts") or {}).keys():
                    excluded_reasons.append(str(reason))
            cumulative_correct = []
            cumulative_cost = []
            running_cost = parent_completion_tokens
            any_correct = False
            discrete_tokens_at_full_k = None
            prefix_tokens_at_full_k = None
            for sample in usable_samples:
                any_correct = any_correct or bool(sample["correct"])
                running_cost += int(sample.get("prefix_completion_tokens", 0))
                running_cost += int(sample["completion_tokens"])
                cumulative_correct.append(any_correct)
                cumulative_cost.append(running_cost)
            if len(usable_samples) >= max_k:
                discrete_tokens_at_full_k = sum(
                    int(sample["completion_tokens"]) for sample in usable_samples[:max_k]
                )
                prefix_tokens_at_full_k = parent_completion_tokens + sum(
                    int(sample.get("prefix_completion_tokens", 0))
                    for sample in usable_samples[:max_k]
                )

            first_correct_k = None
            for index, value in enumerate(cumulative_correct, start=1):
                if value:
                    first_correct_k = index
                    break

            row = {
                "method": method,
                "prompt_index": prompt_index,
                "sample_count": len(samples),
                "usable_sample_count": len(usable_samples),
                "excluded_sample_count": max(len(samples) - len(usable_samples), 0),
                "success": success,
                "error": error,
                "excluded_reasons": "|".join(sorted(set(excluded_reasons))) if excluded_reasons else "",
                "first_correct_k": first_correct_k,
                "parent_completion_tokens": parent_completion_tokens,
                "attempts_used": attempts_used,
                "total_attempt_completion_tokens": total_attempt_completion_tokens,
                "total_attempt_latency_seconds": total_attempt_latency_seconds,
                "discrete_tokens_at_full_k": discrete_tokens_at_full_k,
                "prefix_tokens_at_full_k": prefix_tokens_at_full_k,
            }
            for k in pass_at_ks:
                row[f"pass_at_{k}"] = (
                    bool(cumulative_correct[k - 1]) if len(cumulative_correct) >= k else None
                )
                row[f"cost_at_{k}"] = (
                    cumulative_cost[k - 1] if len(cumulative_cost) >= k else None
                )
            all_method_prompt_rows.append(row)

    row_lookup = {
        (row["method"], int(row["prompt_index"])): row for row in all_method_prompt_rows
    }

    filtered_prompt_rows: List[Dict[str, Any]] = []
    prompt_matching: Dict[int, Dict[str, Any]] = {}
    prompt_level_exclusion_counts: Dict[str, int] = {}
    for prompt_index in prompt_indices:
        usable_counts = {
            method: int(
                row_lookup.get((method, prompt_index), {}).get("usable_sample_count", 0)
            )
            for method in methods
        }
        matched_usable_samples = min(usable_counts.values()) if usable_counts else 0
        eligible_for_any_comparison = matched_usable_samples >= 1
        exclusion_reasons: List[str] = []
        for method in methods:
            method_row = row_lookup.get((method, prompt_index))
            if method_row and method_row.get("excluded_reasons"):
                exclusion_reasons.extend(str(method_row["excluded_reasons"]).split("|"))
        if not eligible_for_any_comparison:
            for method, usable_count in usable_counts.items():
                if usable_count == 0:
                    exclusion_reasons.append(f"{method}_no_usable_samples")
        normalized_exclusion_reasons = sorted({reason for reason in exclusion_reasons if reason})
        for reason in normalized_exclusion_reasons:
            prompt_level_exclusion_counts[reason] = (
                prompt_level_exclusion_counts.get(reason, 0) + 1
            )
        prompt_matching[prompt_index] = {
            "matched_usable_samples": matched_usable_samples,
            "eligible_for_any_comparison": eligible_for_any_comparison,
            "exclusion_reasons": normalized_exclusion_reasons,
        }
        if not eligible_for_any_comparison:
            continue
        for method in methods:
            row = dict(row_lookup[(method, prompt_index)])
            row["matched_usable_samples"] = matched_usable_samples
            row["eligible_for_any_comparison"] = True
            row["prompt_exclusion_reasons"] = "|".join(normalized_exclusion_reasons)
            for k in pass_at_ks:
                if matched_usable_samples < k:
                    row[f"pass_at_{k}"] = None
                    row[f"cost_at_{k}"] = None
            filtered_prompt_rows.append(row)

    summary_rows: List[Dict[str, Any]] = []
    eligible_prompts_by_k: Dict[int, List[int]] = {}
    for method in methods:
        for k in pass_at_ks:
            method_rows = [
                row
                for row in filtered_prompt_rows
                if row["method"] == method and int(row["matched_usable_samples"]) >= k
            ]
            eligible_prompts_by_k[k] = sorted(
                {
                    int(row["prompt_index"])
                    for row in filtered_prompt_rows
                    if int(row["matched_usable_samples"]) >= k
                }
            )
            pass_at_k = None
            avg_cost_tokens = None
            if method_rows:
                pass_at_k = sum(
                    1 for row in method_rows if bool(row[f"pass_at_{k}"])
                ) / len(method_rows)
                avg_cost_tokens = (
                    sum(float(row[f"cost_at_{k}"]) for row in method_rows) / len(method_rows)
                )
            summary_rows.append(
                {
                    "method": method,
                    "k": k,
                    "num_prompts": len(method_rows),
                    "pass_at_k": pass_at_k,
                    "avg_cost_tokens": avg_cost_tokens,
                    "num_failures": sum(1 for row in method_rows if not row["success"]),
                }
            )

    clean_pass_at_ks = [k for k in pass_at_ks if k <= max_k]
    matched_max_k_prompt_indices = sorted(
        {
            int(row["prompt_index"])
            for row in filtered_prompt_rows
            if int(row["matched_usable_samples"]) >= max_k
        }
    )
    clean_summary_rows: List[Dict[str, Any]] = []
    for method in methods:
        method_rows = [
            row
            for row in filtered_prompt_rows
            if row["method"] == method and int(row["matched_usable_samples"]) >= max_k
        ]
        for k in clean_pass_at_ks:
            pass_at_k = None
            avg_cost_tokens = None
            if method_rows:
                pass_at_k = sum(
                    1 for row in method_rows if bool(row.get(f"pass_at_{k}"))
                ) / len(method_rows)
                avg_cost_tokens = (
                    sum(float(row[f"cost_at_{k}"]) for row in method_rows) / len(method_rows)
                )
            clean_summary_rows.append(
                {
                    "method": method,
                    "k": k,
                    "num_prompts": len(method_rows),
                    "pass_at_k": pass_at_k,
                    "avg_cost_tokens": avg_cost_tokens,
                    "num_failures": sum(1 for row in method_rows if not row["success"]),
                }
            )

    retry_statistics: Dict[str, Dict[str, Any]] = {}
    discrete_generation_stats: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        method_attempts = [row for row in attempt_rows if row["method"] == method]
        accepted_attempts = [row for row in method_attempts if row.get("accepted_for_eval")]
        rejected_attempts = [row for row in method_attempts if not row.get("accepted_for_eval")]
        reject_counter = Counter(
            str(row["reject_reason"])
            for row in rejected_attempts
            if row.get("reject_reason")
        )
        retry_statistics[method] = {
            "total_attempts": len(method_attempts),
            "accepted_attempts": len(accepted_attempts),
            "rejected_attempts": len(rejected_attempts),
            "total_completion_tokens": sum(int(row.get("completion_tokens", 0)) for row in method_attempts),
            "accepted_completion_tokens": sum(
                int(row.get("completion_tokens", 0)) for row in accepted_attempts
            ),
            "rejected_completion_tokens": sum(
                int(row.get("completion_tokens", 0)) for row in rejected_attempts
            ),
            "total_latency_seconds": sum(
                float(row.get("latency_seconds", 0.0)) for row in method_attempts
            ),
            "accepted_latency_seconds": sum(
                float(row.get("latency_seconds", 0.0)) for row in accepted_attempts
            ),
            "rejected_latency_seconds": sum(
                float(row.get("latency_seconds", 0.0)) for row in rejected_attempts
            ),
            "reject_reason_counts": dict(reject_counter),
        }
        full_k_rows = [
            row
            for row in all_method_prompt_rows
            if row["method"] == method and int(row["usable_sample_count"]) >= max_k
        ]
        discrete_generation_stats[method] = {
            "prompts_with_full_k": len(full_k_rows),
            "avg_discrete_tokens_per_prompt_full_k": (
                sum(float(row["discrete_tokens_at_full_k"]) for row in full_k_rows) / len(full_k_rows)
                if full_k_rows
                else None
            ),
            "avg_prefix_tokens_per_prompt_full_k": (
                sum(float(row["prefix_tokens_at_full_k"]) for row in full_k_rows) / len(full_k_rows)
                if full_k_rows
                else None
            ),
        }

    method_prompts_with_full_k = {
        method: sum(
            1
            for row in all_method_prompt_rows
            if row["method"] == method and int(row["usable_sample_count"]) >= max_k
        )
        for method in methods
    }

    summary_json = {
        "clean_summary_rows": clean_summary_rows,
        "clean_pass_at_ks": clean_pass_at_ks,
        "summary_rows": summary_rows,
        "per_prompt_rows": filtered_prompt_rows,
        "all_per_prompt_rows": all_method_prompt_rows,
        "eligible_prompts_by_k": {str(k): prompts for k, prompts in eligible_prompts_by_k.items()},
        "prompt_matching": {str(k): v for k, v in prompt_matching.items()},
        "exclusion_counts": prompt_level_exclusion_counts,
        "matched_max_k_prompt_indices": matched_max_k_prompt_indices,
        "coverage": {
            "method_prompts_with_full_k": method_prompts_with_full_k,
            "baseline_prompts_with_full_k": method_prompts_with_full_k.get(
                "baseline_independent", 0
            ),
            "shared_trace_prompts_with_full_k": method_prompts_with_full_k.get(
                "shared_trace_branch_after_prefix", 0
            ),
            "standard_generation_prompts_with_full_k": method_prompts_with_full_k.get(
                "standard_generation_independent", 0
            ),
            "matched_prompts_with_full_k": len(matched_max_k_prompt_indices),
        },
        "retry_statistics": retry_statistics,
        "discrete_generation_stats": discrete_generation_stats,
    }
    return summary_rows, filtered_prompt_rows, summary_json


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary_md(
    path: Path,
    clean_summary_rows: List[Dict[str, Any]],
    detailed_summary_rows: List[Dict[str, Any]],
    pass_at_ks: List[int],
    summary_json: Optional[Dict[str, Any]] = None,
) -> None:
    grouped: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for row in clean_summary_rows:
        grouped.setdefault(row["method"], {})[int(row["k"])] = row

    method_names = {
        "baseline_independent": "Baseline Independent",
        "shared_trace_branch_after_prefix": "Shared Trace After Fixed Prefix",
        "standard_generation_independent": "Standard Generation",
    }
    lines = ["# AIME 2024 Pass@k Comparison", ""]
    if summary_json is not None:
        coverage = summary_json.get("coverage", {})
        if coverage:
            lines.append("## Clean Summary")
            lines.append("")
            lines.append(
                f"- matched prompts with full k={max(pass_at_ks) if pass_at_ks else 0}: {coverage.get('matched_prompts_with_full_k', 0)}"
            )
            lines.append(
                f"- baseline prompts with full k: {coverage.get('baseline_prompts_with_full_k', 0)}"
            )
            lines.append(
                f"- shared-trace prompts with full k: {coverage.get('shared_trace_prompts_with_full_k', 0)}"
            )
            if "standard_generation_prompts_with_full_k" in coverage:
                lines.append(
                    f"- standard-generation prompts with full k: {coverage.get('standard_generation_prompts_with_full_k', 0)}"
                )
            lines.append("")
    lines.append("| Method | " + " | ".join(f"pass@{k}" for k in pass_at_ks) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in pass_at_ks) + " |")
    for method, per_k in grouped.items():
        values = [
            (
                f"{per_k[k]['pass_at_k']:.4f}"
                if k in per_k and per_k[k]["pass_at_k"] is not None
                else ""
            )
            for k in pass_at_ks
        ]
        lines.append("| " + method_names.get(method, method) + " | " + " | ".join(values) + " |")

    lines.append("")
    lines.append("| Method | " + " | ".join(f"cost@{k}" for k in pass_at_ks) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in pass_at_ks) + " |")
    for method, per_k in grouped.items():
        values = [
            (
                f"{per_k[k]['avg_cost_tokens']:.1f}"
                if k in per_k and per_k[k]["avg_cost_tokens"] is not None
                else ""
            )
            for k in pass_at_ks
        ]
        lines.append("| " + method_names.get(method, method) + " | " + " | ".join(values) + " |")
    if summary_json is not None:
        eligible_prompts_by_k = summary_json.get("eligible_prompts_by_k", {})
        exclusion_counts = summary_json.get("exclusion_counts", {})
        retry_statistics = summary_json.get("retry_statistics", {})
        run_timing = summary_json.get("run_timing", {})
        lines.append("")
        lines.append("## Detailed Matched-Per-k Summary")
        lines.append("")
        for method, per_k in grouped.items():
            detailed_rows_by_k = {
                int(row["k"]): row
                for row in detailed_summary_rows
                if row["method"] == method
            }
            values = []
            for k in pass_at_ks:
                detailed_row = detailed_rows_by_k.get(k)
                if detailed_row is None:
                    values.append("")
                else:
                    values.append(str(detailed_row["num_prompts"]))
            lines.append(
                "- "
                + method_names.get(method, method)
                + ": "
                + ", ".join(f"pass@{k} prompts={value or 0}" for k, value in zip(pass_at_ks, values))
            )
        if retry_statistics:
            lines.append("")
            lines.append("## Retry Statistics")
            lines.append("")
            for method, stats in retry_statistics.items():
                lines.append(
                    "- "
                    + method_names.get(method, method)
                    + ": "
                    + f"attempts={stats.get('total_attempts', 0)}, "
                    + f"accepted={stats.get('accepted_attempts', 0)}, "
                    + f"rejected={stats.get('rejected_attempts', 0)}, "
                    + f"tokens={stats.get('total_completion_tokens', 0)}, "
                    + f"rejected_tokens={stats.get('rejected_completion_tokens', 0)}"
                )
                reject_reason_counts = stats.get("reject_reason_counts", {})
                if reject_reason_counts:
                    lines.append(
                        "  reject_reasons="
                        + ", ".join(
                            f"{reason}:{count}"
                            for reason, count in sorted(reject_reason_counts.items())
                        )
                    )
        if run_timing.get("wall_clock_seconds") is not None:
            lines.append("")
            lines.append("## Wall Clock")
            lines.append("")
            lines.append(
                f"- duration: {run_timing.get('wall_clock_hms') or format_wall_clock_seconds(run_timing.get('wall_clock_seconds'))} "
                f"({float(run_timing['wall_clock_seconds']):.1f} seconds)"
            )
            if run_timing.get("started_at"):
                lines.append(f"- started_at: {run_timing['started_at']}")
            if run_timing.get("finished_at"):
                lines.append(f"- finished_at: {run_timing['finished_at']}")
        if eligible_prompts_by_k:
            lines.append("")
            lines.append("## Matched Denominators")
            lines.append("")
            for k in pass_at_ks:
                prompt_count = len(eligible_prompts_by_k.get(str(k), []))
                lines.append(f"- pass@{k}: {prompt_count} matched prompts")
        if exclusion_counts:
            lines.append("")
            lines.append("## Exclusions")
            lines.append("")
            for reason, count in sorted(exclusion_counts.items()):
                lines.append(f"- {reason}: {count}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(
    output_dir: Path,
    summary_rows: List[Dict[str, Any]],
    *,
    passk_filename: str = "passk_curve.png",
    cost_filename: str = "cost_vs_passk.png",
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to write pass@k plots. Install it in the multiplex-thinking environment."
        ) from exc

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault(row["method"], []).append(row)

    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["k"]))

    plt.figure(figsize=(10, 6))
    for method, rows in grouped.items():
        rows = [row for row in rows if row["pass_at_k"] is not None]
        if not rows:
            continue
        plt.plot(
            [int(row["k"]) for row in rows],
            [float(row["pass_at_k"]) for row in rows],
            marker="o",
            label=method,
        )
    plt.xscale("log", base=2)
    plt.xlabel("k")
    plt.ylabel("pass@k")
    plt.title("AIME 2024 pass@k")
    plt.grid(True, which="both", linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / passk_filename, dpi=200)
    plt.close()

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    for method, rows in grouped.items():
        rows = [
            row
            for row in rows
            if row["pass_at_k"] is not None and row["avg_cost_tokens"] is not None
        ]
        if not rows:
            continue
        xs = [float(row["avg_cost_tokens"]) for row in rows]
        ys = [float(row["pass_at_k"]) for row in rows]
        ax.plot(xs, ys, marker="o", label=method)
        for row, x, y in zip(rows, xs, ys):
            ax.annotate(
                f"k={int(row['k'])}",
                (x, y),
                textcoords="offset points",
                xytext=(4, 6),
                fontsize=8,
            )
    plt.xlabel("Average Generated Tokens Per Prompt")
    plt.ylabel("pass@k")
    plt.title("Cost vs pass@k")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / cost_filename, dpi=200)
    plt.close()


def method_display_name(method: str) -> str:
    mapping = {
        "baseline_independent": "Fixed-Prefix Independent",
        "shared_trace_branch_after_prefix": "Shared Trace After Prefix",
        "standard_generation_independent": "Standard Generation",
    }
    return mapping.get(method, method)


def write_overall_overlay_plots(
    root_output_dir: Path,
    prefix_summaries: Dict[int, Dict[str, Any]],
    standard_summary: Optional[Dict[str, Any]],
    pass_at_ks: List[int],
    max_k: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "matplotlib is required to write pass@k plots. Install it in the multiplex-thinking environment."
        ) from exc

    sweep_series: List[Dict[str, Any]] = []
    for reasoning_prefix_tokens, prefix_summary in sorted(prefix_summaries.items()):
        for row in prefix_summary.get("clean_summary_rows", []):
            if int(row["k"]) > max_k:
                continue
            row_copy = dict(row)
            row_copy["reasoning_prefix_tokens"] = reasoning_prefix_tokens
            sweep_series.append(row_copy)
    if standard_summary is not None:
        for row in standard_summary.get("clean_summary_rows", []):
            if int(row["k"]) > max_k:
                continue
            row_copy = dict(row)
            row_copy["reasoning_prefix_tokens"] = 0
            sweep_series.append(row_copy)

    grouped_passk: Dict[str, List[Dict[str, Any]]] = {}
    grouped_cost: Dict[str, List[Dict[str, Any]]] = {}
    for row in sweep_series:
        method = str(row["method"])
        reasoning_prefix_tokens = int(row.get("reasoning_prefix_tokens", 0))
        if method == "standard_generation_independent":
            series_name = method_display_name(method)
        else:
            series_name = f"{method_display_name(method)} · T={reasoning_prefix_tokens}"
        grouped_passk.setdefault(series_name, []).append(row)
        grouped_cost.setdefault(series_name, []).append(row)

    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    for series_name, rows in sorted(grouped_passk.items()):
        rows = [row for row in rows if row.get("pass_at_k") is not None]
        rows.sort(key=lambda row: int(row["k"]))
        if not rows:
            continue
        xs = [int(row["k"]) for row in rows]
        ys = [float(row["pass_at_k"]) for row in rows]
        ax.plot(xs, ys, marker="o", label=series_name)
    ax.set_xscale("log", base=2)
    ax.set_xticks(pass_at_ks)
    ax.set_xticklabels([str(k) for k in pass_at_ks])
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title("Pass@k Overlay Across Reasoning Budgets")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(root_output_dir / "passk_overlay_by_budget.png", dpi=220)
    plt.close()

    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    for series_name, rows in sorted(grouped_cost.items()):
        rows = [
            row for row in rows if row.get("pass_at_k") is not None and row.get("avg_cost_tokens") is not None
        ]
        rows.sort(key=lambda row: int(row["k"]))
        if not rows:
            continue
        xs = [float(row["avg_cost_tokens"]) for row in rows]
        ys = [float(row["pass_at_k"]) for row in rows]
        ax.plot(xs, ys, marker="o", label=series_name)
        for row, x, y in zip(rows, xs, ys):
            ax.annotate(
                f"k={int(row['k'])}",
                (x, y),
                textcoords="offset points",
                xytext=(4, 6),
                fontsize=8,
            )
    ax.set_xlabel("Average Generated Tokens Per Prompt")
    ax.set_ylabel("pass@k")
    ax.set_title("Cost vs pass@k Overlay")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(root_output_dir / "cost_vs_passk_overall.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    for method in ("baseline_independent", "shared_trace_branch_after_prefix"):
        xs: List[int] = []
        ys: List[float] = []
        for reasoning_prefix_tokens, prefix_summary in sorted(prefix_summaries.items()):
            stats = (
                prefix_summary.get("discrete_generation_stats", {}).get(method, {})
            )
            value = stats.get("avg_discrete_tokens_per_prompt_full_k")
            if value is None:
                continue
            xs.append(int(reasoning_prefix_tokens))
            ys.append(float(value))
        if not xs:
            continue
        ax.plot(xs, ys, marker="o", label=method_display_name(method))
    ax.set_xlabel("Reasoning Prefix Tokens")
    ax.set_ylabel(f"Average Discrete Tokens Per Prompt (k={max_k})")
    ax.set_title("Reasoning Budget vs Required Discrete Generation")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(root_output_dir / "reasoning_budget_vs_discrete_generation.png", dpi=220)
    plt.close()


def build_manifest_payload(
    args: argparse.Namespace,
    runtime_config: RuntimeConfig,
    pass_at_ks: List[int],
    examples: List[Example],
    methods: List[str],
    prompt_build_info: PromptBuildInfo,
    multiplex_self_check: Optional[MultiplexSelfCheckResult] = None,
    summary_json: Optional[Dict[str, Any]] = None,
    run_status: str = "running",
    run_failure: Optional[Dict[str, Any]] = None,
    run_timing: Optional[Dict[str, Any]] = None,
    reasoning_prefix_tokens: Optional[int] = None,
    reasoning_prefix_token_values: Optional[List[int]] = None,
) -> Dict[str, Any]:
    git_commit = ""
    try:
        git_commit = subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except Exception:
        git_commit = ""

    server_args = make_server_args(args, runtime_config)
    manifest = {
        "model": args.model,
        "benchmark": args.benchmark,
        "max_k": args.max_k,
        "pass_at_ks": pass_at_ks,
        "methods": methods,
        "seed": args.seed,
        "host": args.host,
        "port": args.port,
        "dp_size": runtime_config.effective_dp_size,
        "tp_size": runtime_config.effective_tp_size,
        "request_batch_size": runtime_config.request_batch_size,
        "requested_dp_size": runtime_config.requested_dp_size,
        "requested_tp_size": runtime_config.requested_tp_size,
        "requested_request_batch_size": runtime_config.requested_request_batch_size,
        "requested_resource_profile": runtime_config.requested_resource_profile,
        "resource_profile": runtime_config.resource_profile,
        "visible_gpu_count": runtime_config.visible_gpu_count,
        "per_gpu_memory_gb": runtime_config.per_gpu_memory_gb,
        "max_new_tokens": args.max_new_tokens,
        "reasoning_prefix_token_values": reasoning_prefix_token_values
        or parse_reasoning_prefix_token_values(args.reasoning_prefix_token_values),
        "checkpoint_matched_prompts_step": args.checkpoint_matched_prompts_step,
        "examples": len(examples),
        "git_commit": git_commit,
        "runtime_config": runtime_config_payload(runtime_config),
        "server_args": serialize_server_args(server_args),
        "assistant_prefill": prompt_build_info.assistant_prefill,
        "prompt_build_mode": prompt_build_info.prompt_build_mode,
        "sampling_defaults": base_sampling_params(
            args.max_new_tokens,
            enable_soft_thinking=True,
        ),
        "fixed_prefix_sampling_defaults": {
            str(prefix_tokens): fixed_prefix_sampling_params(prefix_tokens)
            for prefix_tokens in (
                reasoning_prefix_token_values
                or parse_reasoning_prefix_token_values(args.reasoning_prefix_token_values)
            )
        },
        "child_sampling_defaults": child_sampling_params(
            continuation_token_budget(
                args.max_new_tokens,
                reasoning_prefix_tokens or min(
                    reasoning_prefix_token_values
                    or parse_reasoning_prefix_token_values(args.reasoning_prefix_token_values)
                ),
            )
        ),
        "standard_generation_sampling_defaults": standard_generation_sampling_params(
            args.max_new_tokens
        ),
        "run_status": run_status,
    }
    if reasoning_prefix_tokens is not None:
        manifest["reasoning_prefix_tokens"] = reasoning_prefix_tokens
    if multiplex_self_check is not None:
        manifest["multiplex_self_check"] = asdict(multiplex_self_check)
    if summary_json is not None:
        manifest["summary_metadata"] = {
            "eligible_prompts_by_k": summary_json.get("eligible_prompts_by_k", {}),
            "exclusion_counts": summary_json.get("exclusion_counts", {}),
            "matched_max_k_prompt_indices": summary_json.get("matched_max_k_prompt_indices", []),
            "coverage": summary_json.get("coverage", {}),
        }
    if run_timing is not None:
        manifest["run_timing"] = run_timing
    if run_failure is not None:
        manifest["run_failure"] = run_failure
    return manifest


def artifact_stem(base_name: str, artifact_label: Optional[str] = None) -> str:
    if artifact_label:
        return f"{base_name}_{artifact_label}"
    return base_name


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    runtime_config: RuntimeConfig,
    pass_at_ks: List[int],
    examples: List[Example],
    methods: List[str],
    prompt_build_info: PromptBuildInfo,
    multiplex_self_check: Optional[MultiplexSelfCheckResult] = None,
    summary_json: Optional[Dict[str, Any]] = None,
    run_status: str = "running",
    run_failure: Optional[Dict[str, Any]] = None,
    run_timing: Optional[Dict[str, Any]] = None,
    reasoning_prefix_tokens: Optional[int] = None,
    reasoning_prefix_token_values: Optional[List[int]] = None,
) -> None:
    manifest = build_manifest_payload(
        args,
        runtime_config,
        pass_at_ks,
        examples,
        methods,
        prompt_build_info,
        multiplex_self_check=multiplex_self_check,
        summary_json=summary_json,
        run_status=run_status,
        run_failure=run_failure,
        run_timing=run_timing,
        reasoning_prefix_tokens=reasoning_prefix_tokens,
        reasoning_prefix_token_values=reasoning_prefix_token_values,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_artifacts(
    output_dir: Path,
    sample_rows: List[Dict[str, Any]],
    baseline_prompt_rows: List[Dict[str, Any]],
    parent_rows: List[Dict[str, Any]],
    attempt_rows: List[Dict[str, Any]],
    pass_at_ks: List[int],
    max_k: int,
    run_timing: Optional[Dict[str, Any]] = None,
    artifact_label: Optional[str] = None,
) -> Dict[str, Any]:
    summary_rows, per_prompt_rows, summary_json = compute_summary_tables(
        sample_rows=sample_rows,
        baseline_prompt_rows=baseline_prompt_rows,
        parent_rows=parent_rows,
        attempt_rows=attempt_rows,
        pass_at_ks=pass_at_ks,
        max_k=max_k,
    )
    if run_timing is not None:
        summary_json["run_timing"] = run_timing
    clean_summary_rows = summary_json["clean_summary_rows"]
    write_csv(output_dir / f"{artifact_stem('summary', artifact_label)}.csv", clean_summary_rows)
    write_csv(
        output_dir / f"{artifact_stem('summary_detailed', artifact_label)}.csv",
        summary_rows,
    )
    write_csv(
        output_dir / f"{artifact_stem('per_prompt_correctness', artifact_label)}.csv",
        per_prompt_rows,
    )
    write_summary_md(
        output_dir / f"{artifact_stem('summary', artifact_label)}.md",
        clean_summary_rows,
        summary_rows,
        pass_at_ks,
        summary_json,
    )
    (output_dir / f"{artifact_stem('summary', artifact_label)}.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_plots(
        output_dir,
        clean_summary_rows,
        passk_filename=f"{artifact_stem('passk_curve', artifact_label)}.png",
        cost_filename=f"{artifact_stem('cost_vs_passk', artifact_label)}.png",
    )
    write_plots(
        output_dir,
        summary_rows,
        passk_filename=f"{artifact_stem('passk_curve_detailed', artifact_label)}.png",
        cost_filename=f"{artifact_stem('cost_vs_passk_detailed', artifact_label)}.png",
    )
    return summary_json


def write_checkpoint_artifacts(
    *,
    output_dir: Path,
    sample_rows: List[Dict[str, Any]],
    baseline_prompt_rows: List[Dict[str, Any]],
    parent_rows: List[Dict[str, Any]],
    attempt_rows: List[Dict[str, Any]],
    pass_at_ks: List[int],
    max_k: int,
    checkpoint_step: int,
    run_timing: Optional[Dict[str, Any]],
) -> None:
    if checkpoint_step <= 0:
        return
    _, _, summary_json = compute_summary_tables(
        sample_rows=sample_rows,
        baseline_prompt_rows=baseline_prompt_rows,
        parent_rows=parent_rows,
        attempt_rows=attempt_rows,
        pass_at_ks=pass_at_ks,
        max_k=max_k,
    )
    matched_prompt_indices = summary_json.get("matched_max_k_prompt_indices", [])
    for count in range(checkpoint_step, len(matched_prompt_indices) + 1, checkpoint_step):
        prompt_subset = set(matched_prompt_indices[:count])
        filtered_samples = [
            row for row in sample_rows if int(row["prompt_index"]) in prompt_subset
        ]
        filtered_baseline_prompt_rows = [
            row
            for row in baseline_prompt_rows
            if int(row["prompt_index"]) in prompt_subset
        ]
        filtered_parent_rows = [
            row for row in parent_rows if int(row["prompt_index"]) in prompt_subset
        ]
        filtered_attempt_rows = [
            row for row in attempt_rows if int(row["prompt_index"]) in prompt_subset
        ]
        write_artifacts(
            output_dir,
            filtered_samples,
            filtered_baseline_prompt_rows,
            filtered_parent_rows,
            filtered_attempt_rows,
            pass_at_ks,
            max_k,
            run_timing=run_timing,
            artifact_label=f"checkpoint_{count:03d}",
        )


def prefix_output_dir(root_output_dir: Path, reasoning_prefix_tokens: int) -> Path:
    return root_output_dir / f"prefix_{reasoning_prefix_tokens:04d}"


def standard_generation_output_dir(root_output_dir: Path) -> Path:
    return root_output_dir / "standard_generation"


def write_prefix_sweep_summary(
    root_output_dir: Path,
    prefix_summaries: Dict[int, Dict[str, Any]],
    standard_summary: Optional[Dict[str, Any]],
    pass_at_ks: List[int],
    max_k: int,
    run_timing: Optional[Dict[str, Any]] = None,
) -> None:
    summary_rows: List[Dict[str, Any]] = []
    summary_json: Dict[str, Any] = {
        "reasoning_prefix_token_values": sorted(prefix_summaries),
        "per_prefix": {},
    }
    if standard_summary is not None:
        summary_json["standard_generation"] = {
            "summary_dir": str(standard_generation_output_dir(root_output_dir)),
            "clean_summary_rows": standard_summary.get("clean_summary_rows", []),
            "coverage": standard_summary.get("coverage", {}),
        }
    if run_timing is not None:
        summary_json["run_timing"] = run_timing

    lines = ["# Fixed-Prefix Branching Sweep", ""]
    if standard_summary is not None:
        standard_dir = standard_generation_output_dir(root_output_dir)
        standard_rows = standard_summary.get("clean_summary_rows", [])
        standard_coverage = standard_summary.get("coverage", {})
        for row in standard_rows:
            row_copy = dict(row)
            row_copy["reasoning_prefix_tokens"] = 0
            summary_rows.append(row_copy)
        lines.append("## Standard Generation")
        lines.append("")
        lines.append(f"- summary_dir: {standard_dir}")
        lines.append(
            f"- prompts with full k: {standard_coverage.get('matched_prompts_with_full_k', 0)}"
        )
        lines.append("")
        lines.append("| Method | k | num_prompts | pass@k | avg_cost_tokens |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in standard_rows:
            pass_at_k = "" if row.get("pass_at_k") is None else f"{float(row['pass_at_k']):.4f}"
            avg_cost = "" if row.get("avg_cost_tokens") is None else f"{float(row['avg_cost_tokens']):.1f}"
            lines.append(
                f"| {row['method']} | {row['k']} | {row['num_prompts']} | {pass_at_k} | {avg_cost} |"
            )
        lines.append("")

    for reasoning_prefix_tokens in sorted(prefix_summaries):
        prefix_summary = prefix_summaries[reasoning_prefix_tokens]
        prefix_dir = prefix_output_dir(root_output_dir, reasoning_prefix_tokens)
        clean_rows = prefix_summary.get("clean_summary_rows", [])
        coverage = prefix_summary.get("coverage", {})
        matched_prompts = prefix_summary.get("matched_max_k_prompt_indices", [])
        for row in clean_rows:
            row_copy = dict(row)
            row_copy["reasoning_prefix_tokens"] = reasoning_prefix_tokens
            summary_rows.append(row_copy)
        summary_json["per_prefix"][str(reasoning_prefix_tokens)] = {
            "summary_dir": str(prefix_dir),
            "clean_summary_rows": clean_rows,
            "coverage": coverage,
            "matched_max_k_prompt_indices": matched_prompts,
        }

        lines.append(f"## Prefix {reasoning_prefix_tokens}")
        lines.append("")
        lines.append(f"- summary_dir: {prefix_dir}")
        lines.append(
            f"- matched prompts with full k: {coverage.get('matched_prompts_with_full_k', 0)}"
        )
        lines.append(
            f"- baseline prompts with full k: {coverage.get('baseline_prompts_with_full_k', 0)}"
        )
        lines.append(
            f"- shared-trace prompts with full k: {coverage.get('shared_trace_prompts_with_full_k', 0)}"
        )
        if matched_prompts:
            lines.append(
                "- matched prompt indices: "
                + ", ".join(str(prompt_index) for prompt_index in matched_prompts)
            )
        lines.append("")
        lines.append("| Method | k | num_prompts | pass@k | avg_cost_tokens |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in clean_rows:
            pass_at_k = "" if row.get("pass_at_k") is None else f"{float(row['pass_at_k']):.4f}"
            avg_cost = "" if row.get("avg_cost_tokens") is None else f"{float(row['avg_cost_tokens']):.1f}"
            lines.append(
                f"| {row['method']} | {row['k']} | {row['num_prompts']} | {pass_at_k} | {avg_cost} |"
            )
        lines.append("")

    if run_timing and run_timing.get("wall_clock_seconds") is not None:
        lines.append("## Wall Clock")
        lines.append("")
        lines.append(
            f"- duration: {run_timing.get('wall_clock_hms') or format_wall_clock_seconds(run_timing.get('wall_clock_seconds'))} "
            f"({float(run_timing['wall_clock_seconds']):.1f} seconds)"
        )
        if run_timing.get("started_at"):
            lines.append(f"- started_at: {run_timing['started_at']}")
        if run_timing.get("finished_at"):
            lines.append(f"- finished_at: {run_timing['finished_at']}")
        lines.append("")

    write_csv(root_output_dir / "summary_overall.csv", summary_rows)
    (root_output_dir / "summary_overall.json").write_text(
        json.dumps(summary_json, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (root_output_dir / "summary_overall.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    write_overall_overlay_plots(
        root_output_dir,
        prefix_summaries,
        standard_summary,
        pass_at_ks,
        max_k,
    )


def run_fixed_prefix_experiment(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    runtime_config: RuntimeConfig,
    pass_at_ks: List[int],
    examples: List[Example],
    methods: List[str],
    prompt_build_info: PromptBuildInfo,
    tokenizer: Any,
    client: SGLangRestClient,
    multiplex_self_check: Optional[MultiplexSelfCheckResult],
    reasoning_prefix_tokens: int,
) -> Dict[str, Any]:
    run_started_at_unix = time.time()
    logger = configure_logging(
        output_dir,
        logger_name=f"compare_passk_aime.prefix_{reasoning_prefix_tokens}",
    )
    event_logger = StructuredEventLogger(output_dir / "events.jsonl")
    samples_path = output_dir / "samples.jsonl"
    baseline_prompts_path = output_dir / "baseline_prompts.jsonl"
    parents_path = output_dir / "shared_trace_parents.jsonl"
    attempts_path = output_dir / "attempts.jsonl"
    sample_rows = load_jsonl(samples_path) if args.resume else []
    baseline_prompt_rows = load_jsonl(baseline_prompts_path) if args.resume else []
    parent_rows = load_jsonl(parents_path) if args.resume else []
    attempt_rows = load_jsonl(attempts_path) if args.resume else []
    if not args.resume and (
        samples_path.exists()
        or baseline_prompts_path.exists()
        or parents_path.exists()
        or attempts_path.exists()
    ):
        raise RuntimeError(
            f"{output_dir} already contains result files. Use --resume or a fresh output directory."
        )

    args.current_reasoning_prefix_tokens = reasoning_prefix_tokens
    write_manifest(
        output_dir,
        args,
        runtime_config,
        pass_at_ks,
        examples,
        methods,
        prompt_build_info,
        multiplex_self_check=multiplex_self_check,
        run_status="running",
        run_timing=build_run_timing(run_started_at_unix),
        reasoning_prefix_tokens=reasoning_prefix_tokens,
        reasoning_prefix_token_values=parse_reasoning_prefix_token_values(
            args.reasoning_prefix_token_values
        ),
    )

    completed_pairs = determine_completed_prompts(baseline_prompt_rows, parent_rows)
    for example in examples:
        target_dp_rank = make_target_dp_rank(
            example.prompt_index,
            runtime_config.effective_dp_size,
        )
        if "baseline_independent" in methods and not completed_pairs.get(
            ("baseline_independent", example.prompt_index), False
        ):
            logger.info(
                "[baseline-fixed-prefix] Prompt %s: starting baseline generation for prefix=%s.",
                example.prompt_index,
                reasoning_prefix_tokens,
            )
            baseline_result, baseline_records, baseline_attempts = run_baseline_for_prompt(
                args=args,
                benchmark=args.benchmark,
                client=client,
                example=example,
                target_dp_rank=target_dp_rank,
                logger=logger,
                event_logger=event_logger,
            )
            if baseline_attempts:
                append_records(attempts_path, [asdict(record) for record in baseline_attempts])
                attempt_rows.extend(asdict(record) for record in baseline_attempts)
            append_records(baseline_prompts_path, [asdict(baseline_result)])
            baseline_prompt_rows.append(asdict(baseline_result))
            append_records(samples_path, [asdict(record) for record in baseline_records])
            sample_rows.extend(asdict(record) for record in baseline_records)
            completed_pairs[("baseline_independent", example.prompt_index)] = True
            logger.info(
                "[baseline-fixed-prefix] Prompt %s: produced %s/%s usable continuations.",
                example.prompt_index,
                len(baseline_records),
                args.max_k,
            )

        if "shared_trace_branch_after_prefix" in methods and not completed_pairs.get(
            ("shared_trace_branch_after_prefix", example.prompt_index), False
        ):
            logger.info(
                "[shared-trace-fixed-prefix] Prompt %s: starting shared-prefix branching for prefix=%s.",
                example.prompt_index,
                reasoning_prefix_tokens,
            )
            parent_trace, child_records, shared_attempts = run_shared_trace_for_prompt(
                args=args,
                benchmark=args.benchmark,
                client=client,
                tokenizer=tokenizer,
                example=example,
                target_dp_rank=target_dp_rank,
                logger=logger,
                event_logger=event_logger,
            )
            if shared_attempts:
                append_records(attempts_path, [asdict(record) for record in shared_attempts])
                attempt_rows.extend(asdict(record) for record in shared_attempts)
            append_records(parents_path, [asdict(parent_trace)])
            parent_rows.append(asdict(parent_trace))
            if child_records:
                append_records(samples_path, [asdict(record) for record in child_records])
                sample_rows.extend(asdict(record) for record in child_records)
            completed_pairs[("shared_trace_branch_after_prefix", example.prompt_index)] = True
            logger.info(
                "[shared-trace-fixed-prefix] Prompt %s: parent success=%s, wrote %s child samples.",
                example.prompt_index,
                parent_trace.success,
                len(child_records),
            )

        write_checkpoint_artifacts(
            output_dir=output_dir,
            sample_rows=sample_rows,
            baseline_prompt_rows=baseline_prompt_rows,
            parent_rows=parent_rows,
            attempt_rows=attempt_rows,
            pass_at_ks=pass_at_ks,
            max_k=args.max_k,
            checkpoint_step=args.checkpoint_matched_prompts_step,
            run_timing=build_run_timing(run_started_at_unix, time.time()),
        )

    run_timing = build_run_timing(run_started_at_unix, time.time())
    summary_json = write_artifacts(
        output_dir,
        sample_rows,
        baseline_prompt_rows,
        parent_rows,
        attempt_rows,
        pass_at_ks,
        args.max_k,
        run_timing=run_timing,
    )
    write_manifest(
        output_dir,
        args,
        runtime_config,
        pass_at_ks,
        examples,
        methods,
        prompt_build_info,
        multiplex_self_check=multiplex_self_check,
        summary_json=summary_json,
        run_status="completed",
        run_timing=run_timing,
        reasoning_prefix_tokens=reasoning_prefix_tokens,
        reasoning_prefix_token_values=parse_reasoning_prefix_token_values(
            args.reasoning_prefix_token_values
        ),
    )
    logger.info(
        "[setup] Prefix %s finished in %s (%.1fs). Summary written to %s.",
        reasoning_prefix_tokens,
        run_timing.get("wall_clock_hms"),
        float(run_timing.get("wall_clock_seconds", 0.0)),
        output_dir / "summary.md",
    )
    return summary_json


def run_standard_generation_experiment(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    runtime_config: RuntimeConfig,
    pass_at_ks: List[int],
    examples: List[Example],
    prompt_build_info: PromptBuildInfo,
    client: SGLangRestClient,
) -> Dict[str, Any]:
    run_started_at_unix = time.time()
    logger = configure_logging(output_dir, logger_name="compare_passk_aime.standard_generation")
    event_logger = StructuredEventLogger(output_dir / "events.jsonl")
    samples_path = output_dir / "samples.jsonl"
    prompt_rows_path = output_dir / "baseline_prompts.jsonl"
    attempts_path = output_dir / "attempts.jsonl"
    sample_rows = load_jsonl(samples_path) if args.resume else []
    prompt_rows = load_jsonl(prompt_rows_path) if args.resume else []
    attempt_rows = load_jsonl(attempts_path) if args.resume else []
    if not args.resume and (samples_path.exists() or prompt_rows_path.exists() or attempts_path.exists()):
        raise RuntimeError(
            f"{output_dir} already contains result files. Use --resume or a fresh output directory."
        )

    write_manifest(
        output_dir,
        args,
        runtime_config,
        pass_at_ks,
        examples,
        ["standard_generation_independent"],
        prompt_build_info,
        run_status="running",
        run_timing=build_run_timing(run_started_at_unix),
        reasoning_prefix_token_values=parse_reasoning_prefix_token_values(
            args.reasoning_prefix_token_values
        ),
    )

    completed_pairs = determine_completed_prompts(prompt_rows, [])
    for example in examples:
        target_dp_rank = make_target_dp_rank(example.prompt_index, runtime_config.effective_dp_size)
        if completed_pairs.get(("standard_generation_independent", example.prompt_index), False):
            continue
        prompt_result, sample_records, attempt_records = run_standard_generation_for_prompt(
            args=args,
            benchmark=args.benchmark,
            client=client,
            example=example,
            target_dp_rank=target_dp_rank,
            logger=logger,
            event_logger=event_logger,
        )
        if attempt_records:
            append_records(attempts_path, [asdict(record) for record in attempt_records])
            attempt_rows.extend(asdict(record) for record in attempt_records)
        append_records(prompt_rows_path, [asdict(prompt_result)])
        prompt_rows.append(asdict(prompt_result))
        append_records(samples_path, [asdict(record) for record in sample_records])
        sample_rows.extend(asdict(record) for record in sample_records)
        completed_pairs[("standard_generation_independent", example.prompt_index)] = True
        write_checkpoint_artifacts(
            output_dir=output_dir,
            sample_rows=sample_rows,
            baseline_prompt_rows=prompt_rows,
            parent_rows=[],
            attempt_rows=attempt_rows,
            pass_at_ks=pass_at_ks,
            max_k=args.max_k,
            checkpoint_step=args.checkpoint_matched_prompts_step,
            run_timing=build_run_timing(run_started_at_unix, time.time()),
        )

    run_timing = build_run_timing(run_started_at_unix, time.time())
    summary_json = write_artifacts(
        output_dir,
        sample_rows,
        prompt_rows,
        [],
        attempt_rows,
        pass_at_ks,
        args.max_k,
        run_timing=run_timing,
    )
    write_manifest(
        output_dir,
        args,
        runtime_config,
        pass_at_ks,
        examples,
        ["standard_generation_independent"],
        prompt_build_info,
        summary_json=summary_json,
        run_status="completed",
        run_timing=run_timing,
        reasoning_prefix_token_values=parse_reasoning_prefix_token_values(
            args.reasoning_prefix_token_values
        ),
    )
    logger.info(
        "[setup] Standard generation finished in %s (%.1fs). Summary written to %s.",
        run_timing.get("wall_clock_hms"),
        float(run_timing.get("wall_clock_seconds", 0.0)),
        output_dir / "summary.md",
    )
    return summary_json


def main() -> None:
    run_started_at_unix = time.time()
    args = parse_args()
    ensure_localhost_no_proxy_env()
    output_dir = ensure_output_dir(args)
    logger = configure_logging(output_dir, logger_name="compare_passk_aime.root")
    event_logger = StructuredEventLogger(output_dir / "events.jsonl")
    methods = normalize_methods(args.methods)
    fixed_prefix_methods = [
        method
        for method in methods
        if method in {"baseline_independent", "shared_trace_branch_after_prefix"}
    ]
    run_standard_generation = "standard_generation_independent" in methods
    pass_at_ks = build_pass_at_k_values(args.max_k)
    reasoning_prefix_token_values = parse_reasoning_prefix_token_values(
        args.reasoning_prefix_token_values
    )
    runtime_config = resolve_runtime_config(args)
    args.effective_dp_size = runtime_config.effective_dp_size
    args.effective_tp_size = runtime_config.effective_tp_size
    args.effective_request_batch_size = runtime_config.request_batch_size
    log_runtime_config(logger, runtime_config)
    event_logger.log(
        "setup_runtime_resolved",
        **runtime_config_payload(runtime_config),
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    examples, prompt_build_info = load_examples(tokenizer, args.benchmark, args.max_prompts)
    write_manifest(
        output_dir,
        args,
        runtime_config,
        pass_at_ks,
        examples,
        methods,
        prompt_build_info,
        run_status="initializing",
        run_timing=build_run_timing(run_started_at_unix),
        reasoning_prefix_token_values=reasoning_prefix_token_values,
    )

    server_args = make_server_args(args, runtime_config)
    server_handle = SGLangServerHandle(server_args, args.server_timeout_seconds)
    logger.info("[setup] Starting the local SGLang server on %s:%s.", args.host, args.port)
    event_logger.log(
        "setup_server_start",
        host=args.host,
        port=args.port,
        model=args.model,
        requested_dp_size=runtime_config.requested_dp_size,
        effective_dp_size=runtime_config.effective_dp_size,
        requested_tp_size=runtime_config.requested_tp_size,
        effective_tp_size=runtime_config.effective_tp_size,
        request_batch_size=runtime_config.request_batch_size,
        resource_profile=runtime_config.resource_profile,
    )
    server_handle.start()
    event_logger.log(
        "setup_server_ready",
        host=args.host,
        port=args.port,
        model=args.model,
        effective_dp_size=runtime_config.effective_dp_size,
        effective_tp_size=runtime_config.effective_tp_size,
    )
    client = SGLangRestClient(
        base_url=f"http://{args.host}:{args.port}",
        api_key=args.api_key,
        timeout=args.server_timeout_seconds,
    )

    multiplex_self_check: Optional[MultiplexSelfCheckResult] = None
    prefix_summaries: Dict[int, Dict[str, Any]] = {}
    standard_summary: Optional[Dict[str, Any]] = None
    try:
        multiplex_self_check = verify_multiplex_runtime(
            client=client,
            tokenizer=tokenizer,
            args=args,
            logger=logger,
            event_logger=event_logger,
        )
        write_manifest(
            output_dir,
            args,
            runtime_config,
            pass_at_ks,
            examples,
            methods,
            prompt_build_info,
            multiplex_self_check=multiplex_self_check,
            run_status="running",
            run_timing=build_run_timing(run_started_at_unix),
            reasoning_prefix_token_values=reasoning_prefix_token_values,
        )
        if not multiplex_self_check.success:
            raise RuntimeError(multiplex_self_check.message)

        if run_standard_generation:
            standard_dir = standard_generation_output_dir(output_dir)
            logger.info("[setup] Starting standard-generation baseline in %s.", standard_dir)
            standard_summary = run_standard_generation_experiment(
                args=args,
                output_dir=standard_dir,
                runtime_config=runtime_config,
                pass_at_ks=pass_at_ks,
                examples=examples,
                prompt_build_info=prompt_build_info,
                client=client,
            )
            write_prefix_sweep_summary(
                output_dir,
                prefix_summaries,
                standard_summary,
                pass_at_ks,
                args.max_k,
                run_timing=build_run_timing(run_started_at_unix, time.time()),
            )

        if fixed_prefix_methods:
            for reasoning_prefix_tokens in reasoning_prefix_token_values:
                prefix_dir = prefix_output_dir(output_dir, reasoning_prefix_tokens)
                logger.info(
                    "[setup] Starting fixed-prefix experiment for reasoning_prefix_tokens=%s in %s.",
                    reasoning_prefix_tokens,
                    prefix_dir,
                )
                prefix_summaries[reasoning_prefix_tokens] = run_fixed_prefix_experiment(
                    args=args,
                    output_dir=prefix_dir,
                    runtime_config=runtime_config,
                    pass_at_ks=pass_at_ks,
                    examples=examples,
                    methods=fixed_prefix_methods,
                    prompt_build_info=prompt_build_info,
                    tokenizer=tokenizer,
                    client=client,
                    multiplex_self_check=multiplex_self_check,
                    reasoning_prefix_tokens=reasoning_prefix_tokens,
                )
                write_prefix_sweep_summary(
                    output_dir,
                    prefix_summaries,
                    standard_summary,
                    pass_at_ks,
                    args.max_k,
                    run_timing=build_run_timing(run_started_at_unix, time.time()),
                )

        run_timing = build_run_timing(run_started_at_unix, time.time())
        write_prefix_sweep_summary(
            output_dir,
            prefix_summaries,
            standard_summary,
            pass_at_ks,
            args.max_k,
            run_timing=run_timing,
        )
        write_manifest(
            output_dir,
            args,
            runtime_config,
            pass_at_ks,
            examples,
            methods,
            prompt_build_info,
            multiplex_self_check=multiplex_self_check,
            summary_json={
                "standard_generation": (
                    {
                        "coverage": standard_summary.get("coverage", {}),
                        "matched_max_k_prompt_indices": standard_summary.get(
                            "matched_max_k_prompt_indices", []
                        ),
                    }
                    if standard_summary is not None
                    else None
                ),
                "prefix_sweep": {
                    str(prefix_tokens): {
                        "matched_max_k_prompt_indices": summary.get(
                            "matched_max_k_prompt_indices", []
                        ),
                        "coverage": summary.get("coverage", {}),
                    }
                    for prefix_tokens, summary in prefix_summaries.items()
                }
            },
            run_status="completed",
            run_timing=run_timing,
            reasoning_prefix_token_values=reasoning_prefix_token_values,
        )
        logger.info(
            "[setup] Finished in %s (%.1fs). Summary written to %s.",
            run_timing.get("wall_clock_hms"),
            float(run_timing.get("wall_clock_seconds", 0.0)),
            output_dir / "summary_overall.md",
        )
        event_logger.log(
            "run_complete",
            output_dir=str(output_dir),
            reasoning_prefix_token_values=reasoning_prefix_token_values,
            wall_clock_seconds=run_timing.get("wall_clock_seconds"),
            wall_clock_hms=run_timing.get("wall_clock_hms"),
        )
    except Exception as exc:
        run_timing = build_run_timing(run_started_at_unix, time.time())
        run_failure = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        logger.exception("[setup] Run aborted to preserve fidelity: %s", exc)
        event_logger.log(
            "run_aborted",
            error_type=type(exc).__name__,
            error=str(exc),
            output_dir=str(output_dir),
            wall_clock_seconds=run_timing.get("wall_clock_seconds"),
            wall_clock_hms=run_timing.get("wall_clock_hms"),
        )
        write_manifest(
            output_dir,
            args,
            runtime_config,
            pass_at_ks,
            examples,
            methods,
            prompt_build_info,
            multiplex_self_check=multiplex_self_check,
            run_status="aborted",
            run_failure=run_failure,
            run_timing=run_timing,
            reasoning_prefix_token_values=reasoning_prefix_token_values,
        )
        logger.info(
            "[setup] Run aborted cleanly after %s. Manifest updated at %s.",
            run_timing.get("wall_clock_hms"),
            output_dir / "manifest.json",
        )
    finally:
        client.close()
        server_handle.stop()


if __name__ == "__main__":
    main()
