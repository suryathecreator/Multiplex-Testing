#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import csv
import json
import logging
import math
import multiprocessing as mp
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

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

GENERIC_AIME_PROMPT_TEMPLATE = """Solve the following AIME problem.

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
DEFAULT_REASONING_PREFIX_TOKEN_VALUES = (256, 512, 1024, 2048, 4096, 6144)
DEFAULT_BRANCH_ABLATION_REASONING_PREFIX_TOKENS = 1024
DEFAULT_BRANCH_ABLATION_GROUP_SIZES = (2, 4, 8, 16)
DEFAULT_ADAPTIVE_ABLATION_SHARED_COUNTS = (2, 4, 6, 8, 10, 12, 14, 16)
DEFAULT_ADAPTIVE_CONFIDENCE_THRESHOLD = 0.75
DEFAULT_HYPERPARAM_REASONING_PREFIX_TOKENS = 1024
DEFAULT_HYPERPARAM_TOP_P_VALUES = (0.75, 0.85, 0.90, 0.95, 1.00)
DEFAULT_HYPERPARAM_TEMPERATURE_VALUES = (0.4, 0.6, 0.8, 1.0, 1.2)
DEFAULT_THINKING_TEMPERATURE = 0.8
DEFAULT_THINKING_TOP_P = 0.95
DEFAULT_AFTER_THINKING_TEMPERATURE = 0.6
DEFAULT_AFTER_THINKING_TOP_P = 0.95
EXPERIMENT_MODE_PASSK_SWEEP = "passk_sweep"
EXPERIMENT_MODE_BRANCH_ABLATION = "branch_ablation"
EXPERIMENT_MODE_ADAPTIVE_ABLATION = "adaptive_ablation"
EXPERIMENT_MODE_MEMORY_MATCH_TOPUP = "memory_match_topup"
EXPERIMENT_MODE_MIXED_MEMORY_MATCH_TOPUP = "mixed_memory_match_topup"
EXPERIMENT_MODE_HYPERPARAM_SWEEP = "hyperparam_sweep"
EXPERIMENT_MODE_CHOICES = (
    EXPERIMENT_MODE_PASSK_SWEEP,
    EXPERIMENT_MODE_BRANCH_ABLATION,
    EXPERIMENT_MODE_ADAPTIVE_ABLATION,
    EXPERIMENT_MODE_MEMORY_MATCH_TOPUP,
    EXPERIMENT_MODE_MIXED_MEMORY_MATCH_TOPUP,
    EXPERIMENT_MODE_HYPERPARAM_SWEEP,
)
BENCHMARK_AIME_2024 = "aime2024"
BENCHMARK_DEEPSCALER_AIME_TRAIN = "deepscaler_aime_train"
BENCHMARK_CHOICES = (
    BENCHMARK_AIME_2024,
    BENCHMARK_DEEPSCALER_AIME_TRAIN,
)
AIME_2024_LOCAL_PATH = REPO_ROOT / "deepscaler" / "deepscaler" / "data" / "test" / "aime.json"
DEEPSCALER_AIME_TRAIN_PATH = REPO_ROOT / "deepscaler" / "deepscaler" / "data" / "train" / "aime.json"
DEEPSCALER_AIME_TRAIN_SELECTION_SEED = 4096
DEEPSCALER_AIME_TRAIN_SELECTED_INDICES = (
    8, 18, 19, 34, 40, 42, 46, 54, 58, 75, 78, 80, 87, 89, 120, 123, 130,
    140, 150, 154, 173, 180, 187, 239, 249, 256, 271, 278, 283, 288, 291,
    292, 311, 343, 350, 353, 363, 365, 367, 374, 379, 383, 390, 393, 399,
    403, 426, 428, 432, 453, 459, 462, 466, 467, 506, 509, 523, 527, 550,
    551, 557, 581, 582, 590, 600, 609, 617, 622, 623, 627, 632, 634, 636,
    641, 646, 665, 671, 711, 739, 749, 751, 755, 756, 760, 793, 804, 810,
    811, 812, 814, 831, 847, 865, 872, 910, 921, 922, 944, 954, 967,
)
DEFAULT_MEMORY_MATCH_SHARED_GROUPS = (2, 4, 8, 16, 32)
HYPERPARAM_PHASE_THINKING = "thinking"
HYPERPARAM_PHASE_DISCRETE = "discrete"
THROUGHPUT_PROFILE_SAFE_AUTO = "safe_auto"
THROUGHPUT_PROFILE_SAFE = "safe"
THROUGHPUT_PROFILE_AGGRESSIVE = "aggressive"
THROUGHPUT_PROFILE_CHOICES = (
    THROUGHPUT_PROFILE_SAFE_AUTO,
    THROUGHPUT_PROFILE_SAFE,
    THROUGHPUT_PROFILE_AGGRESSIVE,
)
RANK_SCHEDULER_DYNAMIC = "dynamic"
RANK_SCHEDULER_ROUND_ROBIN = "round_robin"
RANK_SCHEDULER_CHOICES = (
    RANK_SCHEDULER_DYNAMIC,
    RANK_SCHEDULER_ROUND_ROBIN,
)
DEFAULT_PROMPTS_PER_RANK = 1
DEFAULT_FIXED_PREFIX_PROBE_TOKENS = 64
DEFAULT_STANDARD_PROBE_TOKENS = 256
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
COMPACT_JSONL_ENV = "MULTIPLEX_COMPACT_JSONL"


@dataclass
class Example:
    prompt_index: int
    problem_id: str
    problem: str
    answer: str
    prompt_ids: List[int]
    assistant_prefill: str
    prompt_build_mode: str
    metadata: Dict[str, Any] = field(default_factory=dict)


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
    method: str = "shared_trace_branch_after_prefix"
    forced_think_end_tokens: int = 0
    branch_group_index: int = 0


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
    forced_think_end_tokens: int = 0
    branch_group_index: int = 0


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


@dataclass
class SchedulerConfig:
    requested_max_concurrent_prompts: Optional[int]
    max_concurrent_prompts: int
    prompts_per_rank: int
    rank_scheduler: str
    max_prompt_capacity: int
    clamp_messages: List[str] = field(default_factory=list)


@dataclass
class RuntimeProbeResult:
    candidate_name: str
    success: bool
    runtime_config: Dict[str, Any]
    fixed_prefix_probe: Optional[Dict[str, Any]] = None
    standard_generation_probe: Optional[Dict[str, Any]] = None
    total_tokens_per_second: Optional[float] = None
    error: Optional[str] = None
    self_check: Optional[Dict[str, Any]] = None


@dataclass
class FixedPrefixPromptTaskResult:
    prompt_index: int
    target_dp_rank: int
    baseline_result: Optional[BaselinePromptResult] = None
    baseline_records: List[SampleRecord] = field(default_factory=list)
    baseline_attempts: List[AttemptRecord] = field(default_factory=list)
    parent_trace: Optional[ParentTrace] = None
    child_records: List[SampleRecord] = field(default_factory=list)
    shared_attempts: List[AttemptRecord] = field(default_factory=list)
    event_records: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class StandardPromptTaskResult:
    prompt_index: int
    target_dp_rank: int
    prompt_result: Optional[BaselinePromptResult] = None
    sample_records: List[SampleRecord] = field(default_factory=list)
    attempt_records: List[AttemptRecord] = field(default_factory=list)
    event_records: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MemoryMatchPromptTaskResult:
    group_size: int
    topup_group_size: int
    prompt_index: int
    target_dp_rank: int
    prompt_row: Dict[str, Any]
    sample_records: List[SampleRecord] = field(default_factory=list)
    event_records: List[Dict[str, Any]] = field(default_factory=list)


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


class BufferedEventLogger:
    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def log(self, tag: str, **payload: Any) -> None:
        self.records.append({"ts": time.time(), "tag": tag, **payload})


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


def make_rest_client(
    *,
    host: str,
    port: int,
    api_key: Optional[str],
    timeout: int,
) -> "SGLangRestClient":
    return SGLangRestClient(
        base_url=f"http://{host}:{port}",
        api_key=api_key,
        timeout=timeout,
    )


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
        try:
            result = self._post("/open_session", payload)
        except RuntimeError as exc:
            # Resume jobs use deterministic session ids. If a previous attempt left
            # a stale session behind, clear it and retry once before failing.
            if session_id is None or "/open_session failed with status 400" not in str(exc):
                raise
            self.close_session(session_id)
            result = self._post("/open_session", payload)
        if not isinstance(result, str):
            raise RuntimeError(f"Unexpected /open_session response: {result!r}")
        return result

    def close_session(self, session_id: str) -> None:
        try:
            self._post("/close_session", {"session_id": session_id})
        except Exception:
            # Closing is best-effort cleanup. The generated samples remain valid,
            # and aborting here would discard completed prompt work during resume.
            return

    def fork_request(
        self,
        session_id: str,
        parent_rid: str,
        child_count: int,
        child_rids: List[str],
        child_seeds: List[int],
        target_dp_rank: int,
        allow_non_eot_branch: bool = False,
        force_think_end: bool = False,
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
                "force_think_end": force_think_end,
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
    parser.add_argument(
        "--experiment-mode",
        default=EXPERIMENT_MODE_PASSK_SWEEP,
        choices=EXPERIMENT_MODE_CHOICES,
    )
    parser.add_argument("--model", default="Qwen/Qwen3-4B")
    parser.add_argument("--benchmark", default=BENCHMARK_AIME_2024, choices=BENCHMARK_CHOICES)
    parser.add_argument("--max-k", type=int, default=64)
    parser.add_argument(
        "--methods",
        default="baseline,shared_trace,standard_generation",
        help="Comma-separated subset of baseline,shared_trace,standard_generation",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--compact-jsonl",
        action="store_true",
        help=(
            "Write storage-light JSONL checkpoints by dropping full completion text, "
            "generated suffixes, and large fork token/id payloads. Summaries and plots "
            "retain correctness, answer, token-cost, and timing fields."
        ),
    )
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
    parser.add_argument(
        "--throughput-profile",
        default=THROUGHPUT_PROFILE_SAFE_AUTO,
        choices=THROUGHPUT_PROFILE_CHOICES,
    )
    parser.add_argument("--max-concurrent-prompts", type=int, default=None)
    parser.add_argument("--prompts-per-rank", type=int, default=DEFAULT_PROMPTS_PER_RANK)
    parser.add_argument(
        "--rank-scheduler",
        default=RANK_SCHEDULER_DYNAMIC,
        choices=RANK_SCHEDULER_CHOICES,
    )
    parser.add_argument("--capacity-of-str-len", type=int, default=DEFAULT_CAPACITY_OF_STR_LEN)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--server-timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--max-prompts", type=int, default=50)
    parser.add_argument(
        "--prompt-indices",
        default="",
        help=(
            "Optional comma-separated dataset prompt indices to run. This is intended "
            "for narrow resume repairs; --max-prompts still bounds the dataset load."
        ),
    )
    parser.add_argument(
        "--reasoning-prefix-token-values",
        default=",".join(str(value) for value in DEFAULT_REASONING_PREFIX_TOKEN_VALUES),
        help="Comma-separated fixed reasoning-prefix token counts to evaluate, e.g. 256,512,1024,2048,4096,6144.",
    )
    parser.add_argument(
        "--branch-ablation-reasoning-prefix-tokens",
        type=int,
        default=DEFAULT_BRANCH_ABLATION_REASONING_PREFIX_TOKENS,
    )
    parser.add_argument(
        "--branch-ablation-group-sizes",
        default=",".join(str(value) for value in DEFAULT_BRANCH_ABLATION_GROUP_SIZES),
        help="Comma-separated shared-trace group sizes for branch_ablation, e.g. 2,4,8,16.",
    )
    parser.add_argument(
        "--branch-ablation-no-baseline",
        action="store_true",
        help="Run only branch-ablation shared groups, skipping fixed_trace_independent.",
    )
    parser.add_argument(
        "--memory-match-source-root",
        default="",
        help="Repeat-level root containing bundle outputs for memory_match_topup.",
    )
    parser.add_argument(
        "--memory-match-shared-groups",
        default=",".join(str(value) for value in DEFAULT_MEMORY_MATCH_SHARED_GROUPS),
        help="Comma-separated shared group sizes to memory-match against fixed@max_k.",
    )
    parser.add_argument(
        "--memory-match-topup-generator",
        default="fixed",
        choices=("fixed", "shared_group"),
        help=(
            "Generation policy for memory_match_topup: fixed adds independent fixed-prefix "
            "samples; shared_group adds grouped shared generations at the selected group size."
        ),
    )
    parser.add_argument(
        "--memory-match-topup-shared-group-size",
        type=int,
        default=0,
        help=(
            "Shared group size to use for shared_group top-up generation. "
            "When 0, use the base memory-match shared group size."
        ),
    )
    parser.add_argument(
        "--memory-match-checkpoint-family",
        default="",
        choices=("", "shared2", "shared_topup", "mixed_fixed_shared"),
        help="When set, write aggregate checkpoint plots as top-up prompt rows complete.",
    )
    parser.add_argument(
        "--memory-match-checkpoint-before-label",
        default="",
        help="Optional label for the pre-top-up checkpoint point.",
    )
    parser.add_argument(
        "--memory-match-checkpoint-after-label",
        default="",
        help="Optional label for the post-top-up checkpoint point.",
    )
    parser.add_argument(
        "--memory-match-checkpoint-title",
        default="",
        help="Optional title for memory-match checkpoint plots.",
    )
    parser.add_argument(
        "--memory-match-checkpoint-root",
        default="",
        help="Experiment root scanned for checkpoint plots across repeat_XX top-up shards.",
    )
    parser.add_argument(
        "--memory-match-checkpoint-campaign",
        default="",
        help="Top-up campaign directory name under each repeat_XX used for checkpoint plots.",
    )
    parser.add_argument(
        "--memory-match-checkpoint-step",
        type=int,
        default=20,
        help="Write checkpoint plots at each multiple of this many completed prompt rows.",
    )
    parser.add_argument(
        "--adaptive-ablation-shared-counts",
        default=",".join(str(value) for value in DEFAULT_ADAPTIVE_ABLATION_SHARED_COUNTS),
        help=(
            "Comma-separated shared sample counts for adaptive_ablation. Each condition "
            "generates N shared samples, then fixed-prefix top-up samples when confidence is low."
        ),
    )
    parser.add_argument(
        "--adaptive-confidence-threshold",
        type=float,
        default=DEFAULT_ADAPTIVE_CONFIDENCE_THRESHOLD,
        help="Answer-agreement confidence threshold below which adaptive_ablation adds fixed-prefix samples.",
    )
    parser.add_argument(
        "--hyperparam-reasoning-prefix-tokens",
        type=int,
        default=DEFAULT_HYPERPARAM_REASONING_PREFIX_TOKENS,
    )
    parser.add_argument(
        "--hyperparam-top-p-values",
        default=",".join(f"{value:.2f}" for value in DEFAULT_HYPERPARAM_TOP_P_VALUES),
    )
    parser.add_argument(
        "--hyperparam-temperature-values",
        default=",".join(str(value) for value in DEFAULT_HYPERPARAM_TEMPERATURE_VALUES),
    )
    parser.add_argument(
        "--checkpoint-matched-prompts-step",
        type=int,
        default=DEFAULT_CHECKPOINT_MATCHED_PROMPTS_STEP,
        help="Write summary checkpoints after every N prompts where both methods have full usable k data.",
    )
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    parser.add_argument(
        "--disable-cuda-graph",
        action="store_true",
        help="Disable SGLang CUDA graph capture during eval startup.",
    )
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


def parse_positive_int_values(raw_values: str, *, label: str) -> List[int]:
    values: List[int] = []
    seen: set[int] = set()
    for item in raw_values.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError(f"{label} values must be positive integers, got {value}.")
        if value in seen:
            continue
        values.append(value)
        seen.add(value)
    if not values:
        raise ValueError(f"At least one {label} value must be provided.")
    return values


def parse_nonnegative_int_values(raw_values: str, *, label: str) -> List[int]:
    values: List[int] = []
    seen: set[int] = set()
    for item in raw_values.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise ValueError(f"{label} values must be non-negative integers, got {value}.")
        if value in seen:
            continue
        values.append(value)
        seen.add(value)
    return values


def parse_float_values(raw_values: str, *, label: str) -> List[float]:
    values: List[float] = []
    seen: set[float] = set()
    for item in raw_values.split(","):
        item = item.strip()
        if not item:
            continue
        value = float(item)
        if value <= 0.0:
            raise ValueError(f"{label} values must be positive, got {value}.")
        if value in seen:
            continue
        values.append(value)
        seen.add(value)
    if not values:
        raise ValueError(f"At least one {label} value must be provided.")
    return values


def current_sampling_overrides(args: Optional[argparse.Namespace] = None) -> Dict[str, float]:
    if args is None:
        return {}
    return {
        "thinking_temperature": float(
            getattr(args, "current_thinking_temperature", DEFAULT_THINKING_TEMPERATURE)
        ),
        "thinking_top_p": float(
            getattr(args, "current_thinking_top_p", DEFAULT_THINKING_TOP_P)
        ),
        "after_thinking_temperature": float(
            getattr(
                args,
                "current_after_thinking_temperature",
                DEFAULT_AFTER_THINKING_TEMPERATURE,
            )
        ),
        "after_thinking_top_p": float(
            getattr(args, "current_after_thinking_top_p", DEFAULT_AFTER_THINKING_TOP_P)
        ),
    }


def apply_sampling_condition_to_args(
    args: argparse.Namespace,
    *,
    phase: Optional[str] = None,
    parameter_name: Optional[str] = None,
    parameter_value: Optional[float] = None,
) -> None:
    args.current_thinking_temperature = DEFAULT_THINKING_TEMPERATURE
    args.current_thinking_top_p = DEFAULT_THINKING_TOP_P
    args.current_after_thinking_temperature = DEFAULT_AFTER_THINKING_TEMPERATURE
    args.current_after_thinking_top_p = DEFAULT_AFTER_THINKING_TOP_P
    if phase is None or parameter_name is None or parameter_value is None:
        return
    if phase == HYPERPARAM_PHASE_THINKING and parameter_name == "top_p":
        args.current_thinking_top_p = float(parameter_value)
    elif phase == HYPERPARAM_PHASE_DISCRETE and parameter_name == "top_p":
        args.current_after_thinking_top_p = float(parameter_value)
    elif phase == HYPERPARAM_PHASE_THINKING and parameter_name == "temperature":
        args.current_thinking_temperature = float(parameter_value)
    elif phase == HYPERPARAM_PHASE_DISCRETE and parameter_name == "temperature":
        args.current_after_thinking_temperature = float(parameter_value)
    else:
        raise ValueError(f"Unsupported hyperparameter condition: {phase}/{parameter_name}")


def resolve_scheduler_config(
    args: argparse.Namespace,
    runtime_config: RuntimeConfig,
) -> SchedulerConfig:
    prompts_per_rank = int(args.prompts_per_rank)
    if prompts_per_rank < 1:
        raise ValueError("--prompts-per-rank must be >= 1")

    requested_max_concurrent_prompts = (
        None
        if args.max_concurrent_prompts is None
        else int(args.max_concurrent_prompts)
    )
    if requested_max_concurrent_prompts is not None and requested_max_concurrent_prompts < 1:
        raise ValueError("--max-concurrent-prompts must be >= 1 when provided")

    max_prompt_capacity = max(runtime_config.effective_dp_size * prompts_per_rank, 1)
    max_concurrent_prompts = (
        max_prompt_capacity
        if requested_max_concurrent_prompts is None
        else requested_max_concurrent_prompts
    )
    clamp_messages: List[str] = []
    if max_concurrent_prompts > max_prompt_capacity:
        clamp_messages.append(
            f"requested max_concurrent_prompts={max_concurrent_prompts} exceeds prompt capacity "
            f"{max_prompt_capacity}; using {max_prompt_capacity}"
        )
        max_concurrent_prompts = max_prompt_capacity

    return SchedulerConfig(
        requested_max_concurrent_prompts=requested_max_concurrent_prompts,
        max_concurrent_prompts=max_concurrent_prompts,
        prompts_per_rank=prompts_per_rank,
        rank_scheduler=str(args.rank_scheduler),
        max_prompt_capacity=max_prompt_capacity,
        clamp_messages=clamp_messages,
    )


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
        if request_batch_size > safe_cap and args.throughput_profile != THROUGHPUT_PROFILE_AGGRESSIVE:
            clamp_messages.append(
                f"requested request_batch_size={request_batch_size} exceeds the {resource_profile} safety cap of {safe_cap}; using {safe_cap}"
            )
            request_batch_size = safe_cap
        elif request_batch_size > safe_cap:
            clamp_messages.append(
                f"aggressive throughput profile allows request_batch_size={request_batch_size} above the {resource_profile} safety cap of {safe_cap}"
            )
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
    if args.disable_cuda_graph:
        disable_cuda_graph = True
        cuda_graph_max_bs = None
        clamp_messages.append(
            "CUDA graph capture disabled by --disable-cuda-graph for a more robust eval startup"
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


def log_scheduler_config(logger: logging.Logger, scheduler_config: SchedulerConfig) -> None:
    for message in scheduler_config.clamp_messages:
        logger.warning("[setup] %s", message)
    logger.info(
        "[setup] scheduler config: requested_max_concurrent_prompts=%s max_concurrent_prompts=%s prompts_per_rank=%s max_prompt_capacity=%s rank_scheduler=%s",
        scheduler_config.requested_max_concurrent_prompts,
        scheduler_config.max_concurrent_prompts,
        scheduler_config.prompts_per_rank,
        scheduler_config.max_prompt_capacity,
        scheduler_config.rank_scheduler,
    )


def clone_runtime_config(runtime_config: RuntimeConfig, **overrides: Any) -> RuntimeConfig:
    return replace(runtime_config, **overrides)


def runtime_probe_supported(
    args: argparse.Namespace,
    runtime_config: RuntimeConfig,
) -> Tuple[bool, Optional[str]]:
    if args.throughput_profile == THROUGHPUT_PROFILE_SAFE:
        return False, "throughput_profile=safe"
    if runtime_config.resource_profile != RESOURCE_PROFILE_TWO_GPU_SAFE:
        return False, f"resource_profile={runtime_config.resource_profile}"
    if runtime_config.effective_dp_size != 2:
        return False, f"effective_dp_size={runtime_config.effective_dp_size}"
    if runtime_config.attention_backend != "triton":
        return False, f"attention_backend={runtime_config.attention_backend!r}"
    if runtime_config.per_gpu_memory_gb is not None and runtime_config.per_gpu_memory_gb <= LOW_MEMORY_THRESHOLD_GB:
        return False, f"per_gpu_memory_gb={runtime_config.per_gpu_memory_gb}"
    return True, None


def build_runtime_probe_candidates(
    runtime_config: RuntimeConfig,
) -> List[Tuple[str, RuntimeConfig]]:
    requested_batch_size = max(int(runtime_config.request_batch_size), 1)
    safe_current = clone_runtime_config(
        runtime_config,
        request_batch_size=requested_batch_size,
        cuda_graph_max_bs=None,
        chunked_prefill_size=LOW_GPU_CHUNKED_PREFILL_SIZE,
        max_running_requests=LOW_GPU_MAX_RUNNING_REQUESTS,
        attention_backend="triton",
        decode_attention_backend="triton",
        prefill_attention_backend="triton",
        sampling_backend=None,
        disable_cuda_graph=True,
        disable_radix_cache=False,
    )
    cuda_graph_16 = clone_runtime_config(
        safe_current,
        cuda_graph_max_bs=requested_batch_size,
        disable_cuda_graph=False,
    )
    cuda_graph_16_prefill4k = clone_runtime_config(
        cuda_graph_16,
        chunked_prefill_size=4096,
    )
    return [
        ("safe_current", safe_current),
        ("cuda_graph_16", cuda_graph_16),
        ("cuda_graph_16_prefill4k", cuda_graph_16_prefill4k),
    ]


def select_next_rank(
    *,
    scheduler_config: SchedulerConfig,
    rank_active_counts: Dict[int, int],
    rank_completed_counts: Dict[int, int],
    round_robin_cursor: int,
) -> Tuple[Optional[int], int]:
    available_ranks = [
        rank
        for rank, active_count in sorted(rank_active_counts.items())
        if active_count < scheduler_config.prompts_per_rank
    ]
    if not available_ranks:
        return None, round_robin_cursor

    if scheduler_config.rank_scheduler == RANK_SCHEDULER_DYNAMIC:
        rank = min(
            available_ranks,
            key=lambda item: (
                rank_active_counts[item],
                rank_completed_counts.get(item, 0),
                item,
            ),
        )
        return rank, round_robin_cursor

    ordered_ranks = sorted(rank_active_counts)
    if not ordered_ranks:
        return None, round_robin_cursor
    start_index = round_robin_cursor % len(ordered_ranks)
    for offset in range(len(ordered_ranks)):
        rank = ordered_ranks[(start_index + offset) % len(ordered_ranks)]
        if rank_active_counts[rank] < scheduler_config.prompts_per_rank:
            return rank, (ordered_ranks.index(rank) + 1) % len(ordered_ranks)
    return None, round_robin_cursor


def synthetic_probe_prompt_ids(tokenizer: Any) -> List[int]:
    prompt_text = build_prompt("What is 3+5?")
    prompt_ids, _ = build_prefilled_prompt_ids(tokenizer, prompt_text)
    return prompt_ids


def run_fixed_prefix_runtime_probe(
    *,
    client: SGLangRestClient,
    args: argparse.Namespace,
    prompt_ids: List[int],
    target_dp_rank: int,
    batch_size: int,
) -> Dict[str, Any]:
    reasoning_prefix_tokens = DEFAULT_FIXED_PREFIX_PROBE_TOKENS
    decode_budget = DEFAULT_STANDARD_PROBE_TOKENS
    sampling_overrides = current_sampling_overrides(args)
    prefix_params = fixed_prefix_sampling_params(
        reasoning_prefix_tokens,
        sampling_overrides=sampling_overrides,
    )
    continuation_params = child_sampling_params(
        decode_budget,
        sampling_overrides=sampling_overrides,
    )
    session_ids = [
        f"probe-fixed-r{target_dp_rank}-slot{slot_index}-{int(time.time() * 1e6)}"
        for slot_index in range(batch_size)
    ]
    prefix_rids = [f"{session_id}-prefix" for session_id in session_ids]
    continuation_rids = [f"{session_id}-continuation" for session_id in session_ids]
    child_seeds = planned_child_seeds(args.seed, target_dp_rank + 9000, batch_size)
    opened_session_ids: List[str] = []
    total_completion_tokens = 0
    start = time.perf_counter()
    try:
        for session_id in session_ids:
            client.open_session(args.capacity_of_str_len, session_id=session_id)
            opened_session_ids.append(session_id)
        prefix_outputs = normalize_generate_outputs(
            client.generate(
                input_ids=[prompt_ids] * batch_size,
                sampling_params=[dict(prefix_params) for _ in range(batch_size)],
                rid=prefix_rids,
                data_parallel_rank=target_dp_rank,
                session_params=[{"id": session_id} for session_id in session_ids],
            ),
            batch_size,
            "runtime fixed-prefix probe prefix batch",
        )
        continuation_inputs: List[List[int]] = []
        continuation_rids_ready: List[str] = []
        for slot_index, prefix_output in enumerate(prefix_outputs):
            prefix_completion_tokens = int(prefix_output.get("meta_info", {}).get("completion_tokens", 0))
            total_completion_tokens += prefix_completion_tokens
            if not reached_reasoning_prefix(prefix_output, reasoning_prefix_tokens):
                raise RuntimeError("fixed-prefix runtime probe did not reach the requested reasoning prefix")
            fork_info = client.fork_request(
                session_id=session_ids[slot_index],
                parent_rid=prefix_rids[slot_index],
                child_count=1,
                child_rids=[continuation_rids[slot_index]],
                child_seeds=[child_seeds[slot_index]],
                target_dp_rank=target_dp_rank,
                allow_non_eot_branch=True,
                force_think_end=True,
            )
            if not fork_info.get("success", False):
                raise RuntimeError(f"fixed-prefix runtime probe fork failed: {fork_info.get('message')}")
            continuation_inputs.append(list(fork_info["branch_input_ids"]))
            continuation_rids_ready.append(continuation_rids[slot_index])

        continuation_outputs = normalize_generate_outputs(
            client.generate(
                input_ids=continuation_inputs,
                sampling_params=[dict(continuation_params) for _ in continuation_inputs],
                rid=continuation_rids_ready,
                data_parallel_rank=target_dp_rank,
            ),
            len(continuation_inputs),
            "runtime fixed-prefix probe continuation batch",
        )
        total_completion_tokens += sum(
            int(output.get("meta_info", {}).get("completion_tokens", 0))
            for output in continuation_outputs
        )
        elapsed = max(time.perf_counter() - start, 1e-6)
        return {
            "batch_size": batch_size,
            "reasoning_prefix_tokens": reasoning_prefix_tokens,
            "decode_budget": decode_budget,
            "total_completion_tokens": total_completion_tokens,
            "wall_clock_seconds": elapsed,
            "tokens_per_second": total_completion_tokens / elapsed,
            "target_dp_rank": target_dp_rank,
        }
    finally:
        for session_id in reversed(opened_session_ids):
            try:
                client.close_session(session_id)
            except Exception:
                pass


def run_standard_generation_runtime_probe(
    *,
    client: SGLangRestClient,
    prompt_ids: List[int],
    target_dp_rank: int,
    batch_size: int,
) -> Dict[str, Any]:
    params = standard_generation_sampling_params(DEFAULT_STANDARD_PROBE_TOKENS)
    rid_values = [
        f"probe-standard-r{target_dp_rank}-sample{sample_index}-{int(time.time() * 1e6)}"
        for sample_index in range(batch_size)
    ]
    start = time.perf_counter()
    outputs = normalize_generate_outputs(
        client.generate(
            input_ids=[prompt_ids] * batch_size,
            sampling_params=[dict(params) for _ in range(batch_size)],
            rid=rid_values,
            data_parallel_rank=target_dp_rank,
        ),
        batch_size,
        "runtime standard-generation probe batch",
    )
    total_completion_tokens = sum(
        int(output.get("meta_info", {}).get("completion_tokens", 0))
        for output in outputs
    )
    elapsed = max(time.perf_counter() - start, 1e-6)
    return {
        "batch_size": batch_size,
        "decode_budget": DEFAULT_STANDARD_PROBE_TOKENS,
        "total_completion_tokens": total_completion_tokens,
        "wall_clock_seconds": elapsed,
        "tokens_per_second": total_completion_tokens / elapsed,
        "target_dp_rank": target_dp_rank,
    }


def autotune_runtime_config(
    *,
    args: argparse.Namespace,
    runtime_config: RuntimeConfig,
    tokenizer: Any,
    logger: logging.Logger,
    event_logger: StructuredEventLogger,
) -> Tuple[RuntimeConfig, Dict[str, Any]]:
    enabled, skipped_reason = runtime_probe_supported(args, runtime_config)
    metadata: Dict[str, Any] = {
        "enabled": enabled,
        "mode": str(args.throughput_profile),
        "skipped_reason": skipped_reason,
        "selected_candidate": "safe_current",
        "candidates": [],
        "skipped_candidates": [],
    }
    if not enabled:
        return runtime_config, metadata

    prompt_ids = synthetic_probe_prompt_ids(tokenizer)
    batch_size = max(int(runtime_config.request_batch_size), 1)
    candidate_results: List[RuntimeProbeResult] = []
    selected_runtime_config = runtime_config
    best_tokens_per_second = -1.0
    successful_candidate_name = "safe_current"
    probe_candidates = build_runtime_probe_candidates(runtime_config)
    if runtime_config.nvcc_path is None:
        skipped_candidates = [name for name, _ in probe_candidates if name != "safe_current"]
        if skipped_candidates:
            logger.info(
                "[setup] skipping CUDA-graph runtime probe candidates on the no-nvcc path: %s",
                ",".join(skipped_candidates),
            )
            for candidate_name in skipped_candidates:
                event_logger.log(
                    "runtime_probe_candidate_skipped",
                    candidate_name=candidate_name,
                    reason="nvcc_unavailable_on_runtime_path",
                )
        metadata["skipped_candidates"] = skipped_candidates
        probe_candidates = [candidate for candidate in probe_candidates if candidate[0] == "safe_current"]

    for candidate_name, candidate_runtime_config in probe_candidates:
        server_handle: Optional[SGLangServerHandle] = None
        client: Optional[SGLangRestClient] = None
        try:
            logger.info("[setup] probing runtime candidate %s", candidate_name)
            event_logger.log(
                "runtime_probe_candidate_start",
                candidate_name=candidate_name,
                runtime_config=runtime_config_payload(candidate_runtime_config),
            )
            server_args = make_server_args(args, candidate_runtime_config)
            server_handle = SGLangServerHandle(server_args, args.server_timeout_seconds)
            server_handle.start()
            client = make_rest_client(
                host=args.host,
                port=args.port,
                api_key=args.api_key,
                timeout=args.server_timeout_seconds,
            )
            self_check = verify_multiplex_runtime(
                client=client,
                tokenizer=tokenizer,
                args=args,
                logger=logger,
                event_logger=event_logger,
            )
            if not self_check.success:
                raise RuntimeError(self_check.message)

            fixed_rank = 0
            standard_rank = min(candidate_runtime_config.effective_dp_size - 1, 1)
            fixed_probe = run_fixed_prefix_runtime_probe(
                client=client,
                args=args,
                prompt_ids=prompt_ids,
                target_dp_rank=fixed_rank,
                batch_size=batch_size,
            )
            standard_probe = run_standard_generation_runtime_probe(
                client=client,
                prompt_ids=prompt_ids,
                target_dp_rank=standard_rank,
                batch_size=batch_size,
            )
            total_tokens = (
                float(fixed_probe["total_completion_tokens"])
                + float(standard_probe["total_completion_tokens"])
            )
            total_time = (
                float(fixed_probe["wall_clock_seconds"])
                + float(standard_probe["wall_clock_seconds"])
            )
            total_tokens_per_second = total_tokens / max(total_time, 1e-6)
            result = RuntimeProbeResult(
                candidate_name=candidate_name,
                success=True,
                runtime_config=runtime_config_payload(candidate_runtime_config),
                fixed_prefix_probe=fixed_probe,
                standard_generation_probe=standard_probe,
                total_tokens_per_second=total_tokens_per_second,
                self_check=asdict(self_check),
            )
            candidate_results.append(result)
            event_logger.log(
                "runtime_probe_candidate_success",
                candidate_name=candidate_name,
                total_tokens_per_second=total_tokens_per_second,
                fixed_prefix_probe=fixed_probe,
                standard_generation_probe=standard_probe,
            )
            if total_tokens_per_second > best_tokens_per_second:
                best_tokens_per_second = total_tokens_per_second
                selected_runtime_config = candidate_runtime_config
                successful_candidate_name = candidate_name
        except Exception as exc:
            logger.warning("[setup] runtime probe candidate %s failed: %s", candidate_name, exc)
            candidate_results.append(
                RuntimeProbeResult(
                    candidate_name=candidate_name,
                    success=False,
                    runtime_config=runtime_config_payload(candidate_runtime_config),
                    error=str(exc),
                )
            )
            event_logger.log(
                "runtime_probe_candidate_failed",
                candidate_name=candidate_name,
                error=str(exc),
            )
        finally:
            if client is not None:
                client.close()
            if server_handle is not None:
                server_handle.stop()

    metadata["candidates"] = [asdict(result) for result in candidate_results]
    metadata["selected_candidate"] = successful_candidate_name
    metadata["selected_total_tokens_per_second"] = (
        best_tokens_per_second if best_tokens_per_second >= 0.0 else None
    )
    if best_tokens_per_second < 0.0:
        raise RuntimeError("All runtime probe candidates failed; refusing to continue.")
    logger.info(
        "[setup] selected runtime candidate %s (%.2f tok/s during startup probe)",
        successful_candidate_name,
        best_tokens_per_second,
    )
    event_logger.log(
        "runtime_probe_selected",
        candidate_name=successful_candidate_name,
        total_tokens_per_second=best_tokens_per_second,
    )
    return selected_runtime_config, metadata


def build_pass_at_k_values(max_k: int) -> List[int]:
    values = [value for value in DEFAULT_PASS_AT_KS if value <= max_k]
    if max_k not in values:
        values.append(max_k)
    return values


def build_prompt(problem: str, benchmark: str = BENCHMARK_AIME_2024) -> str:
    template = (
        AIME_PROMPT_TEMPLATE
        if benchmark == BENCHMARK_AIME_2024
        else GENERIC_AIME_PROMPT_TEMPLATE
    )
    return template.format(problem=problem.strip())


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


def aime_number_label(value: Any) -> str:
    try:
        number = int(value)
    except Exception:
        return str(value or "unknown")
    return {1: "I", 2: "II"}.get(number, str(number))


def normalize_deepscaler_aime_problem_id(row: Dict[str, Any], idx: int) -> str:
    year = row.get("year", "unknown")
    aime_label = aime_number_label(row.get("aime_number"))
    try:
        problem_number = int(row.get("problem_number"))
        problem_label = f"{problem_number:02d}"
    except Exception:
        problem_label = str(row.get("problem_number") or idx)
    return f"aime-train-{year}-{aime_label}-{problem_label}"


def load_examples(
    tokenizer: Any,
    benchmark: str,
    max_prompts: Optional[int],
) -> Tuple[List[Example], PromptBuildInfo]:
    if benchmark not in BENCHMARK_CHOICES:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    if benchmark == BENCHMARK_AIME_2024:
        if AIME_2024_LOCAL_PATH.exists():
            dataset = json.loads(AIME_2024_LOCAL_PATH.read_text(encoding="utf-8"))
        else:
            from datasets import load_dataset

            dataset = list(load_dataset("Maxwell-Jia/AIME_2024", split="train"))
        source_indices = list(range(len(dataset)))
    else:
        if not DEEPSCALER_AIME_TRAIN_PATH.exists():
            raise FileNotFoundError(f"Missing Deepscaler AIME train data: {DEEPSCALER_AIME_TRAIN_PATH}")
        dataset = json.loads(DEEPSCALER_AIME_TRAIN_PATH.read_text(encoding="utf-8"))
        source_indices = list(DEEPSCALER_AIME_TRAIN_SELECTED_INDICES)
        missing = [index for index in source_indices if index < 0 or index >= len(dataset)]
        if missing:
            raise ValueError(f"Selected AIME-train source indices out of range: {missing}")
    if max_prompts is not None:
        source_indices = source_indices[:max_prompts]

    examples: List[Example] = []
    prompt_build_info: Optional[PromptBuildInfo] = None
    for idx in source_indices:
        row = dataset[idx]
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
        prompt_text = build_prompt(problem_text, benchmark=benchmark)
        prompt_ids, row_prompt_build_info = build_prefilled_prompt_ids(tokenizer, prompt_text)
        if prompt_build_info is None:
            prompt_build_info = row_prompt_build_info
        metadata: Dict[str, Any] = {
            "source_index": idx,
            "source_dataset": (
                str(AIME_2024_LOCAL_PATH.relative_to(REPO_ROOT))
                if benchmark == BENCHMARK_AIME_2024 and AIME_2024_LOCAL_PATH.exists()
                else "Maxwell-Jia/AIME_2024"
                if benchmark == BENCHMARK_AIME_2024
                else str(DEEPSCALER_AIME_TRAIN_PATH.relative_to(REPO_ROOT))
            ),
        }
        if benchmark == BENCHMARK_AIME_2024 and AIME_2024_LOCAL_PATH.exists():
            metadata.update(
                {
                    "year": row.get("year"),
                    "aime_number": row.get("aime_number"),
                    "problem_number": row.get("problem_number"),
                    "difficulty": row.get("difficulty"),
                    "solution": row.get("solution"),
                }
            )
        if benchmark == BENCHMARK_DEEPSCALER_AIME_TRAIN:
            metadata.update(
                {
                    "year": row.get("year"),
                    "aime_number": row.get("aime_number"),
                    "problem_number": row.get("problem_number"),
                    "difficulty": row.get("difficulty"),
                    "solution": row.get("solution"),
                    "selection_seed": DEEPSCALER_AIME_TRAIN_SELECTION_SEED,
                }
            )
        examples.append(
            Example(
                prompt_index=idx,
                problem_id=(
                    normalize_deepscaler_aime_problem_id(row, idx)
                    if benchmark == BENCHMARK_DEEPSCALER_AIME_TRAIN
                    else normalize_problem_id(row, idx)
                ),
                problem=problem_text,
                answer=answer_text,
                prompt_ids=prompt_ids,
                assistant_prefill=row_prompt_build_info.assistant_prefill,
                prompt_build_mode=row_prompt_build_info.prompt_build_mode,
                metadata=metadata,
            )
        )
    if prompt_build_info is None:
        prompt_build_info = PromptBuildInfo(
            assistant_prefill=ASSISTANT_THINK_PREFILL,
            prompt_build_mode="native_generation_prompt",
        )
    return examples, prompt_build_info


def selection_manifest_rows(examples: List[Example]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for order_index, example in enumerate(examples):
        metadata = dict(example.metadata or {})
        row = {
            "order_index": order_index,
            "prompt_index": example.prompt_index,
            "problem_id": example.problem_id,
            "answer": example.answer,
            "problem": example.problem,
        }
        for key in (
            "source_index",
            "source_dataset",
            "year",
            "aime_number",
            "problem_number",
            "difficulty",
            "selection_seed",
        ):
            if key in metadata:
                row[key] = metadata[key]
        rows.append(row)
    return rows


def write_selection_manifest(output_dir: Path, args: argparse.Namespace, examples: List[Example]) -> None:
    payload = {
        "benchmark": args.benchmark,
        "selection_seed": (
            DEEPSCALER_AIME_TRAIN_SELECTION_SEED
            if args.benchmark == BENCHMARK_DEEPSCALER_AIME_TRAIN
            else None
        ),
        "selected_count": len(examples),
        "selected_source_indices": [
            int((example.metadata or {}).get("source_index", example.prompt_index))
            for example in examples
        ],
        "examples": selection_manifest_rows(examples),
    }
    (output_dir / "selection_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def make_server_args(args: argparse.Namespace, runtime_config: RuntimeConfig) -> Any:
    from sglang.srt.server_args import ServerArgs

    return ServerArgs(
        model_path=args.model,
        tokenizer_path=args.model,
        served_model_name=args.model,
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        # We already run an explicit startup health check, runtime probe, and
        # multiplex self-check after launch. SGLang's extra background warmup
        # has proven brittle on these 2-GPU nodes, where it can hang during a
        # tiny synthetic request and trip the worker watchdog before the real
        # experiment begins.
        skip_server_warmup=True,
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
    sampling_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    sampling_overrides = sampling_overrides or {}
    thinking_temperature = float(
        sampling_overrides.get("thinking_temperature", DEFAULT_THINKING_TEMPERATURE)
    )
    thinking_top_p = float(
        sampling_overrides.get("thinking_top_p", DEFAULT_THINKING_TOP_P)
    )
    after_thinking_temperature = float(
        sampling_overrides.get(
            "after_thinking_temperature",
            DEFAULT_AFTER_THINKING_TEMPERATURE,
        )
    )
    after_thinking_top_p = float(
        sampling_overrides.get("after_thinking_top_p", DEFAULT_AFTER_THINKING_TOP_P)
    )
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


def fixed_prefix_sampling_params(
    reasoning_prefix_tokens: int,
    *,
    sampling_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    params = base_sampling_params(
        max_new_tokens=reasoning_prefix_tokens,
        enable_soft_thinking=True,
        sampling_overrides=sampling_overrides,
    )
    params["think_end_str"] = FIXED_PREFIX_NEVER_SWITCH_THINK_END
    params["ignore_eos"] = True
    params.pop("stop", None)
    return params


def child_sampling_params(
    max_new_tokens: int,
    *,
    sampling_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    baseline_params = base_sampling_params(
        max_new_tokens=max_new_tokens,
        enable_soft_thinking=True,
        sampling_overrides=sampling_overrides,
    )
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": baseline_params["after_thinking_temperature"],
        "top_p": baseline_params["after_thinking_top_p"],
        "top_k": baseline_params["after_thinking_top_k"],
        "min_p": baseline_params["after_thinking_min_p"],
        "custom_params": {"__disable_soft_thinking__": True},
    }


def standard_generation_sampling_params(
    max_new_tokens: int,
    *,
    sampling_overrides: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    # The plain-generation baseline uses the same non-multiplex decoding settings
    # that the fixed-prefix methods use after the multiplex reasoning prefix.
    return child_sampling_params(max_new_tokens, sampling_overrides=sampling_overrides)


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
    forced_think_end_tokens: int = 0,
    branch_group_index: int = 0,
    reasoning_prefix_tokens: Optional[int] = None,
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
        parent_completion_tokens=(
            parent_trace.completion_tokens + parent_trace.forced_think_end_tokens
            if parent_trace
            else 0
        ),
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
            reasoning_prefix_tokens
            if reasoning_prefix_tokens is not None
            else (
                parent_trace.reasoning_prefix_tokens
                if parent_trace is not None
                else prefix_completion_tokens
            )
        ),
        forced_think_end_tokens=forced_think_end_tokens,
        branch_group_index=branch_group_index,
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


def truthy_env_flag(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def compact_jsonl_enabled() -> bool:
    return truthy_env_flag(os.environ.get(COMPACT_JSONL_ENV))


def compact_jsonl_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Keep resumable accounting fields while dropping bulky free-text payloads."""
    compacted = dict(record)

    for key in ("text", "generated_suffix", "shared_trace_text"):
        if key in compacted:
            compacted[key] = ""

    # score_debug often contains large candidate text copies. The scalar
    # score_reason/extracted_answer fields are enough for summaries.
    if "score_debug" in compacted:
        compacted["score_debug"] = None

    for key in ("branch_input_ids", "cacheable_input_ids", "uncached_tail_input_ids"):
        if key in compacted:
            compacted[key] = []

    # Be defensive around event records from error paths.
    for key in ("response", "output_text", "output_preview"):
        value = compacted.get(key)
        if isinstance(value, str) and len(value) > 500:
            compacted[key] = value[:500] + "...[compact-jsonl-truncated]"

    compacted["compact_jsonl"] = True
    return compacted


def append_records(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    writer = JsonlWriter(path)
    should_compact = compact_jsonl_enabled()
    for record in records:
        writer.append(compact_jsonl_record(record) if should_compact else record)


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


def load_summary_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summary_has_observed_results(summary_json: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(summary_json, dict):
        return False
    for row in summary_json.get("clean_summary_rows", []):
        if int(row.get("num_prompts", 0) or 0) > 0:
            return True
        if row.get("pass_at_k") is not None:
            return True
    return False


def infer_available_pass_at_ks(sample_rows: List[Dict[str, Any]]) -> List[int]:
    if not sample_rows:
        return [1]
    counts: Counter[Tuple[str, int]] = Counter()
    for row in sample_rows:
        method = str(row.get("method", ""))
        prompt_index = int(row.get("prompt_index", -1))
        counts[(method, prompt_index)] += 1
    max_available_k = max(counts.values(), default=1)
    return build_pass_at_k_values(max_available_k)


def summarize_existing_output_dir(
    output_dir: Path,
    *,
    pass_at_ks: List[int],
    max_k: int,
    includes_shared_trace: bool,
) -> Optional[Dict[str, Any]]:
    samples_path = output_dir / "samples.jsonl"
    baseline_prompts_path = output_dir / "baseline_prompts.jsonl"
    attempts_path = output_dir / "attempts.jsonl"
    parents_path = output_dir / "shared_trace_parents.jsonl"
    if not samples_path.exists() and not baseline_prompts_path.exists() and not parents_path.exists():
        return None

    persisted_summary = load_summary_json(output_dir / "summary.json")
    manifest = load_summary_json(output_dir / "manifest.json") or {}
    if summary_has_observed_results(persisted_summary):
        if isinstance(manifest.get("run_timing"), dict):
            persisted_summary["run_timing"] = manifest["run_timing"]
        return persisted_summary

    sample_rows = load_jsonl(samples_path)
    baseline_prompt_rows = load_jsonl(baseline_prompts_path)
    attempt_rows = load_jsonl(attempts_path)
    parent_rows = load_jsonl(parents_path) if includes_shared_trace else []
    inferred_pass_at_ks = infer_available_pass_at_ks(sample_rows)
    inferred_max_k = max(inferred_pass_at_ks)
    _, _, summary_json = compute_summary_tables(
        sample_rows=sample_rows,
        baseline_prompt_rows=baseline_prompt_rows,
        parent_rows=parent_rows,
        attempt_rows=attempt_rows,
        pass_at_ks=inferred_pass_at_ks,
        max_k=inferred_max_k,
    )
    run_timing = manifest.get("run_timing")
    if isinstance(run_timing, dict):
        summary_json["run_timing"] = run_timing
    return summary_json


def load_existing_sweep_summaries(
    root_output_dir: Path,
    *,
    pass_at_ks: List[int],
    max_k: int,
) -> Tuple[Dict[int, Dict[str, Any]], Optional[Dict[str, Any]]]:
    prefix_summaries: Dict[int, Dict[str, Any]] = {}
    for prefix_dir in sorted(root_output_dir.glob("prefix_*")):
        if not prefix_dir.is_dir():
            continue
        suffix = prefix_dir.name.removeprefix("prefix_")
        if not suffix.isdigit():
            continue
        summary_json = summarize_existing_output_dir(
            prefix_dir,
            pass_at_ks=pass_at_ks,
            max_k=max_k,
            includes_shared_trace=True,
        )
        if summary_json is None:
            continue
        prefix_summaries[int(suffix)] = summary_json

    standard_dir = standard_generation_output_dir(root_output_dir)
    standard_summary = summarize_existing_output_dir(
        standard_dir,
        pass_at_ks=pass_at_ks,
        max_k=max_k,
        includes_shared_trace=False,
    )
    return prefix_summaries, standard_summary


def determine_completed_prompts(
    baseline_prompt_rows: List[Dict[str, Any]],
    parent_rows: List[Dict[str, Any]],
) -> Dict[Tuple[str, int], bool]:
    def row_has_full_k(row: Dict[str, Any]) -> bool:
        if "usable_for_eval" in row and "required_sample_count" not in row:
            return bool(row.get("success", False)) and bool(row.get("usable_for_eval", False))
        required = int(row.get("required_sample_count") or row.get("usable_sample_count") or 0)
        usable = int(row.get("usable_sample_count") or 0)
        return bool(row.get("success", False)) and required > 0 and usable >= required

    completed: Dict[Tuple[str, int], bool] = {}
    for row in baseline_prompt_rows:
        if not row_has_full_k(row):
            continue
        method = str(row.get("method", "baseline_independent"))
        completed[(method, int(row["prompt_index"]))] = True
    for row in parent_rows:
        if not row_has_full_k(row):
            continue
        completed[
            (
                str(row.get("method", "shared_trace_branch_after_prefix")),
                int(row["prompt_index"]),
            )
        ] = True
    return completed


def make_scheduler_metadata(
    *,
    scheduler_config: SchedulerConfig,
    rank_completed_counts: Dict[int, int],
    observed_max_in_flight_prompts: int,
) -> Dict[str, Any]:
    return {
        "rank_scheduler": scheduler_config.rank_scheduler,
        "max_concurrent_prompts": scheduler_config.max_concurrent_prompts,
        "prompts_per_rank": scheduler_config.prompts_per_rank,
        "max_prompt_capacity": scheduler_config.max_prompt_capacity,
        "rank_completed_prompt_counts": {
            str(rank): int(count) for rank, count in sorted(rank_completed_counts.items())
        },
        "observed_max_in_flight_prompts": observed_max_in_flight_prompts,
    }


def run_fixed_prefix_prompt_task(
    *,
    args: argparse.Namespace,
    benchmark: str,
    host: str,
    port: int,
    api_key: Optional[str],
    timeout: int,
    example: Example,
    target_dp_rank: int,
    logger: logging.Logger,
    run_baseline: bool,
    run_shared_trace: bool,
    existing_baseline_sample_indices: Optional[set[int]] = None,
) -> FixedPrefixPromptTaskResult:
    client = make_rest_client(host=host, port=port, api_key=api_key, timeout=timeout)
    event_logger = BufferedEventLogger()
    baseline_result: Optional[BaselinePromptResult] = None
    baseline_records: List[SampleRecord] = []
    baseline_attempts: List[AttemptRecord] = []
    parent_trace: Optional[ParentTrace] = None
    child_records: List[SampleRecord] = []
    shared_attempts: List[AttemptRecord] = []
    try:
        if run_baseline:
            logger.info(
                "[baseline-fixed-prefix] Prompt %s: starting baseline generation for prefix=%s.",
                example.prompt_index,
                int(args.current_reasoning_prefix_tokens),
            )
            baseline_result, baseline_records, baseline_attempts = run_baseline_for_prompt(
                args=args,
                benchmark=benchmark,
                client=client,
                example=example,
                target_dp_rank=target_dp_rank,
                logger=logger,
                event_logger=event_logger,
                existing_sample_indices=existing_baseline_sample_indices,
            )
            logger.info(
                "[baseline-fixed-prefix] Prompt %s: produced %s/%s usable continuations.",
                example.prompt_index,
                len(baseline_records),
                args.max_k,
            )

        if run_shared_trace:
            logger.info(
                "[shared-trace-fixed-prefix] Prompt %s: starting shared-prefix branching for prefix=%s.",
                example.prompt_index,
                int(args.current_reasoning_prefix_tokens),
            )
            parent_trace, child_records, shared_attempts = run_shared_trace_for_prompt(
                args=args,
                benchmark=benchmark,
                client=client,
                tokenizer=None,
                example=example,
                target_dp_rank=target_dp_rank,
                logger=logger,
                event_logger=event_logger,
            )
            logger.info(
                "[shared-trace-fixed-prefix] Prompt %s: parent success=%s, wrote %s child samples.",
                example.prompt_index,
                parent_trace.success if parent_trace is not None else False,
                len(child_records),
            )
        return FixedPrefixPromptTaskResult(
            prompt_index=example.prompt_index,
            target_dp_rank=target_dp_rank,
            baseline_result=baseline_result,
            baseline_records=baseline_records,
            baseline_attempts=baseline_attempts,
            parent_trace=parent_trace,
            child_records=child_records,
            shared_attempts=shared_attempts,
            event_records=list(event_logger.records),
        )
    finally:
        client.close()


def run_standard_generation_prompt_task(
    *,
    args: argparse.Namespace,
    benchmark: str,
    host: str,
    port: int,
    api_key: Optional[str],
    timeout: int,
    example: Example,
    target_dp_rank: int,
    logger: logging.Logger,
) -> StandardPromptTaskResult:
    client = make_rest_client(host=host, port=port, api_key=api_key, timeout=timeout)
    event_logger = BufferedEventLogger()
    try:
        prompt_result, sample_records, attempt_records = run_standard_generation_for_prompt(
            args=args,
            benchmark=benchmark,
            client=client,
            example=example,
            target_dp_rank=target_dp_rank,
            logger=logger,
            event_logger=event_logger,
        )
        return StandardPromptTaskResult(
            prompt_index=example.prompt_index,
            target_dp_rank=target_dp_rank,
            prompt_result=prompt_result,
            sample_records=sample_records,
            attempt_records=attempt_records,
            event_records=list(event_logger.records),
        )
    finally:
        client.close()


def run_prompt_tasks_concurrently(
    *,
    task_items: List[Dict[str, Any]],
    scheduler_config: SchedulerConfig,
    effective_dp_size: int,
    submit_task: Any,
    handle_result: Any,
    logger: logging.Logger,
    event_logger: StructuredEventLogger,
) -> Dict[str, Any]:
    pending_items: Deque[Dict[str, Any]] = deque(task_items)
    rank_active_counts = {rank: 0 for rank in range(effective_dp_size)}
    rank_completed_counts = {rank: 0 for rank in range(effective_dp_size)}
    round_robin_cursor = 0
    observed_max_in_flight_prompts = 0

    with ThreadPoolExecutor(max_workers=scheduler_config.max_concurrent_prompts) as executor:
        in_flight: Dict[Future, Tuple[int, int]] = {}

        def maybe_submit_more() -> None:
            nonlocal round_robin_cursor, observed_max_in_flight_prompts
            while pending_items and len(in_flight) < scheduler_config.max_concurrent_prompts:
                selected_rank, round_robin_cursor = select_next_rank(
                    scheduler_config=scheduler_config,
                    rank_active_counts=rank_active_counts,
                    rank_completed_counts=rank_completed_counts,
                    round_robin_cursor=round_robin_cursor,
                )
                if selected_rank is None:
                    return
                item = pending_items.popleft()
                prompt_index = int(item["example"].prompt_index)
                logger.info(
                    "[scheduler] Dispatching prompt %s to DP rank %s.",
                    prompt_index,
                    selected_rank,
                )
                event_logger.log(
                    "prompt_dispatched",
                    prompt_index=prompt_index,
                    target_dp_rank=selected_rank,
                    rank_scheduler=scheduler_config.rank_scheduler,
                    pending_prompts=len(pending_items),
                    in_flight_prompts=len(in_flight),
                )
                future = executor.submit(submit_task, item, selected_rank)
                in_flight[future] = (prompt_index, selected_rank)
                rank_active_counts[selected_rank] += 1
                observed_max_in_flight_prompts = max(
                    observed_max_in_flight_prompts,
                    len(in_flight),
                )

        maybe_submit_more()
        while in_flight:
            done, _ = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
            for future in done:
                prompt_index, target_dp_rank = in_flight.pop(future)
                rank_active_counts[target_dp_rank] -= 1
                try:
                    result = future.result()
                except Exception:
                    for pending_future in in_flight:
                        pending_future.cancel()
                    raise
                handle_result(result)
                rank_completed_counts[target_dp_rank] += 1
                event_logger.log(
                    "prompt_completed",
                    prompt_index=prompt_index,
                    target_dp_rank=target_dp_rank,
                    pending_prompts=len(pending_items),
                    in_flight_prompts=len(in_flight),
                )
            maybe_submit_more()

    return make_scheduler_metadata(
        scheduler_config=scheduler_config,
        rank_completed_counts=rank_completed_counts,
        observed_max_in_flight_prompts=observed_max_in_flight_prompts,
    )


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


def usable_sample_indices_for_prompt(
    sample_rows: Iterable[Dict[str, Any]],
    *,
    method: str,
    prompt_index: int,
) -> set[int]:
    return {
        int(row["sample_index"])
        for row in sample_rows
        if str(row.get("method")) == method
        and int(row.get("prompt_index", -1)) == prompt_index
        and sample_row_usable_for_eval(row)
    }


def run_baseline_for_prompt(
    *,
    args: argparse.Namespace,
    benchmark: str,
    client: SGLangRestClient,
    example: Example,
    target_dp_rank: int,
    logger: logging.Logger,
    event_logger: StructuredEventLogger,
    existing_sample_indices: Optional[set[int]] = None,
) -> Tuple[BaselinePromptResult, List[SampleRecord], List[AttemptRecord]]:
    reasoning_prefix_tokens = int(args.current_reasoning_prefix_tokens)
    decode_budget = continuation_token_budget(args.max_new_tokens, reasoning_prefix_tokens)
    existing_sample_indices = existing_sample_indices or set()
    repair_slot_indices = [
        slot_index for slot_index in range(args.max_k) if slot_index not in existing_sample_indices
    ]
    logger.info(
        "[baseline-fixed-prefix] Prompt %s: generating %s/%s missing independent traces with %s multiplex prefix tokens on DP rank %s, then switching to discrete decoding for %s tokens.",
        example.prompt_index,
        len(repair_slot_indices),
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
        existing_usable_sample_count=len(existing_sample_indices),
        repair_slot_indices=repair_slot_indices,
        reasoning_prefix_tokens=reasoning_prefix_tokens,
        decode_max_new_tokens=decode_budget,
    )
    sampling_overrides = current_sampling_overrides(args)
    prefix_params = fixed_prefix_sampling_params(
        reasoning_prefix_tokens,
        sampling_overrides=sampling_overrides,
    )
    continuation_params = child_sampling_params(
        decode_budget,
        sampling_overrides=sampling_overrides,
    )
    continuation_seeds = planned_child_seeds(args.seed, example.prompt_index, args.max_k)
    slot_statuses: List[Dict[str, Any]] = [
        {
            "slot_index": slot_index,
            "prefix_completion_tokens": 0,
            "prefix_latency_seconds": 0.0,
            "prefix_finish_reason": None,
            "prefix_reached": False,
            "forced_think_end_tokens": 0,
            "continuation_completion_tokens": 0,
            "continuation_latency_seconds": 0.0,
            "accepted_for_eval": False,
            "failure_reason": None,
        }
        for slot_index in range(args.max_k)
    ]
    for slot_index in existing_sample_indices:
        if 0 <= slot_index < len(slot_statuses):
            slot_statuses[slot_index]["accepted_for_eval"] = True
    accepted_records: List[SampleRecord] = []
    session_id_by_slot = {
        slot_index: f"baseline-p{example.prompt_index}-t{reasoning_prefix_tokens}-slot{slot_index}"
        for slot_index in repair_slot_indices
    }
    prefix_rid_by_slot = {
        slot_index: f"{session_id}-prefix"
        for slot_index, session_id in session_id_by_slot.items()
    }
    continuation_rid_by_slot = {
        slot_index: f"{session_id}-continuation"
        for slot_index, session_id in session_id_by_slot.items()
    }
    opened_session_ids: List[str] = []
    continuation_jobs: List[Dict[str, Any]] = []
    try:
        for session_id in session_id_by_slot.values():
            client.open_session(args.capacity_of_str_len, session_id=session_id)
            opened_session_ids.append(session_id)
        event_logger.log(
            "baseline_prefix_batch_start",
            prompt_index=example.prompt_index,
            target_dp_rank=target_dp_rank,
            reasoning_prefix_tokens=reasoning_prefix_tokens,
            batch_size=len(repair_slot_indices),
            repair_slot_indices=repair_slot_indices,
        )
        prefix_outputs = []
        if repair_slot_indices:
            prefix_outputs = normalize_generate_outputs(
                client.generate(
                    input_ids=[example.prompt_ids] * len(repair_slot_indices),
                    sampling_params=[dict(prefix_params) for _ in repair_slot_indices],
                    rid=[prefix_rid_by_slot[slot_index] for slot_index in repair_slot_indices],
                    data_parallel_rank=target_dp_rank,
                    session_params=[
                        {"id": session_id_by_slot[slot_index]}
                        for slot_index in repair_slot_indices
                    ],
                ),
                len(repair_slot_indices),
                "baseline prefix batch",
            )
        event_logger.log(
            "baseline_prefix_batch_done",
            prompt_index=example.prompt_index,
            target_dp_rank=target_dp_rank,
            reasoning_prefix_tokens=reasoning_prefix_tokens,
            batch_size=len(repair_slot_indices),
        )

        for batch_index, prefix_output in enumerate(prefix_outputs):
            slot_index = repair_slot_indices[batch_index]
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
                rid=prefix_rid_by_slot[slot_index],
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
                session_id=session_id_by_slot[slot_index],
                parent_rid=prefix_rid_by_slot[slot_index],
                child_count=1,
                child_rids=[continuation_rid_by_slot[slot_index]],
                child_seeds=[continuation_seeds[slot_index]],
                target_dp_rank=target_dp_rank,
                allow_non_eot_branch=True,
                force_think_end=True,
            )
            if not fork_info.get("success", False):
                slot_status["failure_reason"] = "fork_failed"
                event_logger.log(
                    "baseline_prefix_fork_failed",
                    prompt_index=example.prompt_index,
                    slot_index=slot_index,
                    rid=prefix_rid_by_slot[slot_index],
                    target_dp_rank=target_dp_rank,
                    reasoning_prefix_tokens=reasoning_prefix_tokens,
                    message=fork_info.get("message", "fork_request failed"),
                )
                continue
            forced_think_end_tokens = int(
                fork_info.get("forced_think_end_token_count", 0)
            )
            slot_status["forced_think_end_tokens"] = forced_think_end_tokens
            branch_prefix_text = (
                prefix_text + THINK_END_TAG
                if forced_think_end_tokens and THINK_END_TAG not in prefix_text
                else prefix_text
            )
            continuation_jobs.append(
                {
                    "slot_index": slot_index,
                    "input_ids": fork_info["branch_input_ids"],
                    "rid": continuation_rid_by_slot[slot_index],
                    "planned_seed": continuation_seeds[slot_index],
                    "prefix_text": branch_prefix_text,
                    "prefix_completion_tokens": prefix_completion_tokens
                    + forced_think_end_tokens,
                    "forced_think_end_tokens": forced_think_end_tokens,
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
                    forced_think_end_tokens=int(job["forced_think_end_tokens"]),
                    reasoning_prefix_tokens=reasoning_prefix_tokens,
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
    combined_usable_count = len(
        set(existing_sample_indices)
        | {int(record.sample_index) for record in accepted_records}
    )
    success = combined_usable_count >= args.max_k
    missing_slots = [
        int(status["slot_index"])
        for status in slot_statuses
        if int(status["slot_index"]) not in existing_sample_indices
        and not status["accepted_for_eval"]
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
        usable_sample_count=combined_usable_count,
        attempts_used=len(repair_slot_indices),
        total_completion_tokens_spent=sum(
            int(status["prefix_completion_tokens"])
            + int(status["forced_think_end_tokens"])
            + int(status["continuation_completion_tokens"])
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
    params = standard_generation_sampling_params(
        args.max_new_tokens,
        sampling_overrides=current_sampling_overrides(args),
    )
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
    sampling_overrides = current_sampling_overrides(args)
    parent_params = fixed_prefix_sampling_params(
        reasoning_prefix_tokens,
        sampling_overrides=sampling_overrides,
    )
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
            force_think_end=True,
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
            forced_think_end_tokens=int(fork_info.get("forced_think_end_token_count", 0)),
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

        forced_think_end_tokens = int(fork_info.get("forced_think_end_token_count", 0))
        branch_parent_text = (
            parent_text + THINK_END_TAG
            if forced_think_end_tokens and THINK_END_TAG not in parent_text
            else parent_text
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
            shared_trace_text=branch_parent_text,
            branch_input_ids=fork_info["branch_input_ids"],
            cacheable_input_ids=fork_info["cacheable_input_ids"],
            uncached_tail_input_ids=fork_info["uncached_tail_input_ids"],
            cacheable_token_count=int(fork_info.get("cacheable_token_count", 0)),
            eot_token_id=None,
            eot_output_index=-1,
            prompt_token_count=int(parent_meta.get("prompt_tokens", 0)),
            response_token_count=parent_completion_tokens + forced_think_end_tokens,
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
            total_completion_tokens_spent=parent_completion_tokens + forced_think_end_tokens,
            total_latency_seconds_spent=parent_latency_seconds,
            reject_reason_counts={},
            accepted_attempt_index=None,
            reasoning_prefix_tokens=reasoning_prefix_tokens,
            forced_think_end_tokens=forced_think_end_tokens,
        )

        child_records: List[SampleRecord] = []
        params = child_sampling_params(
            decode_budget,
            sampling_overrides=sampling_overrides,
        )
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
                    forced_think_end_tokens=forced_think_end_tokens if sample_index == 0 else 0,
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
    grouped_sample_slots: Dict[Tuple[str, int], Dict[int, Dict[str, Any]]] = {}
    for row in sample_rows:
        key = (row["method"], int(row["prompt_index"]))
        slot_index = int(row["sample_index"])
        # Resume repair may append a corrected prompt after a partial prompt. Keep the
        # newest record for each sample slot so repaired runs do not double count.
        grouped_sample_slots.setdefault(key, {})[slot_index] = row
    grouped_samples: Dict[Tuple[str, int], List[Dict[str, Any]]] = {
        key: list(slot_rows.values()) for key, slot_rows in grouped_sample_slots.items()
    }

    grouped_attempts: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for row in attempt_rows:
        key = (row["method"], int(row["prompt_index"]))
        grouped_attempts.setdefault(key, []).append(row)

    prompt_result_by_key = {
        (str(row.get("method", "baseline_independent")), int(row["prompt_index"])): row
        for row in baseline_prompt_rows
    }
    parent_by_key = {
        (
            str(row.get("method", "shared_trace_branch_after_prefix")),
            int(row["prompt_index"]),
            int(row.get("branch_group_index", 0)),
        ): row
        for row in parent_rows
    }

    all_method_prompt_rows: List[Dict[str, Any]] = []
    prompt_indices = sorted(
        {
            int(row["prompt_index"]) for row in sample_rows
        }
        | {int(row["prompt_index"]) for row in baseline_prompt_rows}
        | {int(row["prompt_index"]) for row in parent_rows}
        | {int(row["prompt_index"]) for row in attempt_rows}
    )
    methods = sorted(
        {str(row["method"]) for row in sample_rows}
        | {str(row.get("method", "baseline_independent")) for row in baseline_prompt_rows}
        | (
            {
                str(row.get("method", "shared_trace_branch_after_prefix"))
                for row in parent_rows
            }
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
            parent = parent_by_key.get((method, prompt_index, 0))
            prompt_result = prompt_result_by_key.get((method, prompt_index))
            parent_completion_tokens = (
                int(parent["completion_tokens"])
                + int(parent.get("forced_think_end_tokens", 0))
                if method.startswith("shared_trace") and parent
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
            elif method.startswith("shared_trace") and parent is not None:
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
    group_match = re.fullmatch(r"shared_trace_group_(\d+)", method)
    if group_match:
        return f"Shared Trace Every {group_match.group(1)}"
    adaptive_match = re.fullmatch(r"adaptive_shared_(\d+)_fixed_topup", method)
    if adaptive_match:
        shared_count = int(adaptive_match.group(1))
        return f"Adaptive Shared@{shared_count}"
    mapping = {
        "baseline_independent": "Fixed-Prefix Independent",
        "shared_trace_branch_after_prefix": "Shared Trace After Prefix",
        "standard_generation_independent": "Standard Generation",
    }
    return mapping.get(method, method)


def legacy_series_name(method: str, reasoning_prefix_tokens: int) -> str:
    if method == "standard_generation_independent":
        return "standard generation"
    short_name = "baseline" if method == "baseline_independent" else "shared"
    return f"{short_name} (T={reasoning_prefix_tokens})"


def method_style(method: str) -> Dict[str, Any]:
    if method.startswith("shared_trace_group_"):
        return {
            "linestyle": "--",
            "marker": "s",
            "linewidth": 2.1,
            "markersize": 7,
        }
    if method == "standard_generation_independent":
        return {
            "linestyle": "-.",
            "marker": "^",
            "linewidth": 2.3,
            "markersize": 7,
            "color": "#2ca02c",
        }
    if method == "baseline_independent":
        return {
            "linestyle": "-",
            "marker": "o",
            "linewidth": 2.1,
            "markersize": 7,
        }
    return {
        "linestyle": "--",
        "marker": "s",
        "linewidth": 2.1,
        "markersize": 7,
    }


def reasoning_budget_color(reasoning_prefix_tokens: int) -> str:
    mapping = {
        256: "#9467bd",
        512: "#1f77b4",
        1024: "#ff7f0e",
        2048: "#d62728",
        4096: "#8c564b",
    }
    return mapping.get(reasoning_prefix_tokens, "#7f7f7f")


def hex_to_rgb(color: str) -> Tuple[float, float, float]:
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) / 255.0 for index in (0, 2, 4))


def rgb_to_hex(rgb: Tuple[float, float, float]) -> str:
    return "#" + "".join(f"{max(0, min(255, round(channel * 255))):02x}" for channel in rgb)


def blend_with_white(color: str, blend: float) -> str:
    base = hex_to_rgb(color)
    white = (1.0, 1.0, 1.0)
    mixed = tuple((1.0 - blend) * channel + blend * white_channel for channel, white_channel in zip(base, white))
    return rgb_to_hex(mixed)


def series_style(method: str, reasoning_prefix_tokens: int) -> Dict[str, Any]:
    style = dict(method_style(method))
    if method != "standard_generation_independent":
        style["color"] = reasoning_budget_color(reasoning_prefix_tokens)
    return style


def family_series_style(method: str, reasoning_prefix_tokens: int) -> Dict[str, Any]:
    if method == "standard_generation_independent":
        return dict(method_style(method))
    base_color = reasoning_budget_color(reasoning_prefix_tokens)
    style = dict(method_style(method))
    if method == "baseline_independent":
        style["color"] = blend_with_white(base_color, 0.08)
        style["linewidth"] = 2.4
        style["markersize"] = 7.5
    else:
        style["color"] = blend_with_white(base_color, 0.38)
        style["linewidth"] = 2.6
        style["markersize"] = 7.5
    return style


def passk_overlay_dir(root_output_dir: Path) -> Path:
    path = root_output_dir / "passk_overlays"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cost_vs_passk_dir(root_output_dir: Path) -> Path:
    path = root_output_dir / "cost_vs_passk"
    path.mkdir(parents=True, exist_ok=True)
    return path


def reasoning_budget_dir(root_output_dir: Path) -> Path:
    path = root_output_dir / "reasoning_budget"
    path.mkdir(parents=True, exist_ok=True)
    return path


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
    passk_dir = passk_overlay_dir(root_output_dir)
    cost_dir = cost_vs_passk_dir(root_output_dir)
    reasoning_dir = reasoning_budget_dir(root_output_dir)

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
    legacy_grouped_passk: Dict[str, List[Dict[str, Any]]] = {}
    legacy_grouped_cost: Dict[str, List[Dict[str, Any]]] = {}
    for row in sweep_series:
        method = str(row["method"])
        reasoning_prefix_tokens = int(row.get("reasoning_prefix_tokens", 0))
        if method == "standard_generation_independent":
            series_name = method_display_name(method)
        else:
            series_name = f"{method_display_name(method)} · T={reasoning_prefix_tokens}"
        legacy_name = legacy_series_name(method, reasoning_prefix_tokens)
        grouped_passk.setdefault(series_name, []).append(row)
        grouped_cost.setdefault(series_name, []).append(row)
        legacy_grouped_passk.setdefault(legacy_name, []).append(row)
        legacy_grouped_cost.setdefault(legacy_name, []).append(row)

    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    for series_name, rows in sorted(grouped_passk.items()):
        rows = [row for row in rows if row.get("pass_at_k") is not None]
        rows.sort(key=lambda row: int(row["k"]))
        if not rows:
            continue
        xs = [int(row["k"]) for row in rows]
        ys = [float(row["pass_at_k"]) for row in rows]
        exemplar = rows[0]
        ax.plot(
            xs,
            ys,
            label=series_name,
            **series_style(str(exemplar["method"]), int(exemplar.get("reasoning_prefix_tokens", 0))),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(pass_at_ks)
    ax.set_xticklabels([str(k) for k in pass_at_ks])
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title("Pass@k Overlay Across Reasoning Budgets")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(passk_dir / "passk_overlay_by_budget.png", dpi=220)
    plt.close()

    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    for series_name, rows in sorted(grouped_passk.items()):
        rows = [row for row in rows if row.get("pass_at_k") is not None]
        rows.sort(key=lambda row: int(row["k"]))
        if not rows:
            continue
        xs = [int(row["k"]) for row in rows]
        ys = [float(row["pass_at_k"]) for row in rows]
        exemplar = rows[0]
        ax.plot(
            xs,
            ys,
            label=series_name,
            **series_style(str(exemplar["method"]), int(exemplar.get("reasoning_prefix_tokens", 0))),
        )
        for row, x, y in zip(rows, xs, ys):
            ax.annotate(
                f"k={int(row['k'])}",
                (x, y),
                textcoords="offset points",
                xytext=(4, 6),
                fontsize=8,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks(pass_at_ks)
    ax.set_xticklabels([str(k) for k in pass_at_ks])
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title("Pass@k Overlay Across Reasoning Budgets")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(passk_dir / "passk_overlay_by_budget_with_k_labels.png", dpi=220)
    plt.close()

    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    for series_name, rows in sorted(legacy_grouped_passk.items()):
        rows = [row for row in rows if row.get("pass_at_k") is not None]
        rows.sort(key=lambda row: int(row["k"]))
        if not rows:
            continue
        xs = [int(row["k"]) for row in rows]
        ys = [float(row["pass_at_k"]) for row in rows]
        exemplar = rows[0]
        ax.plot(
            xs,
            ys,
            label=series_name,
            **series_style(str(exemplar["method"]), int(exemplar.get("reasoning_prefix_tokens", 0))),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(pass_at_ks)
    ax.set_xticklabels([str(k) for k in pass_at_ks])
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title("pass@k Overlay")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(passk_dir / "passk_overlay.png", dpi=220)
    plt.savefig(passk_dir / "passk_overlay_by_branching_point.png", dpi=220)
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
        exemplar = rows[0]
        ax.plot(
            xs,
            ys,
            label=series_name,
            **series_style(str(exemplar["method"]), int(exemplar.get("reasoning_prefix_tokens", 0))),
        )
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
    plt.savefig(cost_dir / "cost_vs_passk_overall.png", dpi=220)
    plt.close()

    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    for series_name, rows in sorted(legacy_grouped_cost.items()):
        rows = [
            row for row in rows if row.get("pass_at_k") is not None and row.get("avg_cost_tokens") is not None
        ]
        rows.sort(key=lambda row: int(row["k"]))
        if not rows:
            continue
        xs = [float(row["avg_cost_tokens"]) for row in rows]
        ys = [float(row["pass_at_k"]) for row in rows]
        exemplar = rows[0]
        ax.plot(
            xs,
            ys,
            label=series_name,
            **series_style(str(exemplar["method"]), int(exemplar.get("reasoning_prefix_tokens", 0))),
        )
    ax.set_xlabel("Average Generated Tokens Per Prompt")
    ax.set_ylabel("pass@k")
    ax.set_title("Cost vs pass@k")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(cost_dir / "cost_vs_passk_overall_unlabeled.png", dpi=220)
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
        ax.plot(xs, ys, label=method_display_name(method), **method_style(method))
    ax.set_xlabel("Reasoning Prefix Tokens")
    ax.set_ylabel(f"Average Discrete Tokens Per Prompt (k={max_k})")
    ax.set_title("Reasoning Budget vs Required Discrete Generation")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(reasoning_dir / "reasoning_budget_vs_discrete_generation.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    if standard_summary is not None:
        standard_rows = [
            row
            for row in standard_summary.get("clean_summary_rows", [])
            if row.get("pass_at_k") is not None and row.get("avg_cost_tokens") is not None
        ]
        if standard_rows:
            standard_rows.sort(key=lambda row: int(row["k"]))
            final_row = standard_rows[-1]
            ax.plot(
                [0],
                [float(final_row["avg_cost_tokens"])],
                label=method_display_name("standard_generation_independent"),
                **method_style("standard_generation_independent"),
            )
            standard_line = ax.lines[-1]
            standard_line.set_linestyle("None")
    for method in ("baseline_independent", "shared_trace_branch_after_prefix"):
        xs = []
        ys = []
        for reasoning_prefix_tokens, prefix_summary in sorted(prefix_summaries.items()):
            rows = [
                row
                for row in prefix_summary.get("clean_summary_rows", [])
                if row.get("method") == method
                and row.get("pass_at_k") is not None
                and row.get("avg_cost_tokens") is not None
            ]
            if not rows:
                continue
            rows.sort(key=lambda row: int(row["k"]))
            final_row = rows[-1]
            xs.append(int(reasoning_prefix_tokens))
            ys.append(float(final_row["avg_cost_tokens"]))
        if xs:
            ax.plot(xs, ys, label=method_display_name(method), **method_style(method))
    ax.set_xlabel("Reasoning Budget (tokens)")
    ax.set_ylabel("Average Total Generated Tokens Per Prompt")
    ax.set_title("Reasoning Budget vs Required Total Generation")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(reasoning_dir / "reasoning_budget_vs_required_total_generation.png", dpi=220)
    plt.close()

    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    for series_name, rows in sorted(grouped_passk.items()):
        rows = [row for row in rows if row.get("pass_at_k") is not None]
        rows.sort(key=lambda row: int(row["k"]))
        if not rows:
            continue
        exemplar = rows[0]
        xs = [int(row["k"]) for row in rows]
        ys = [float(row["pass_at_k"]) for row in rows]
        ax.plot(
            xs,
            ys,
            label=series_name,
            **series_style(str(exemplar["method"]), int(exemplar.get("reasoning_prefix_tokens", 0))),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(pass_at_ks)
    ax.set_xticklabels([str(k) for k in pass_at_ks])
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title("Pass@k Overlay Across Reasoning Budgets")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(passk_dir / "passk_overlay_by_budget_styled.png", dpi=220)
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
        exemplar = rows[0]
        xs = [float(row["avg_cost_tokens"]) for row in rows]
        ys = [float(row["pass_at_k"]) for row in rows]
        ax.plot(
            xs,
            ys,
            label=series_name,
            **series_style(str(exemplar["method"]), int(exemplar.get("reasoning_prefix_tokens", 0))),
        )
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
    plt.savefig(cost_dir / "cost_vs_passk_overall_styled.png", dpi=220)
    plt.close()

    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    if standard_summary is not None:
        standard_rows = [
            row
            for row in standard_summary.get("clean_summary_rows", [])
            if row.get("pass_at_k") is not None and row.get("avg_cost_tokens") is not None
        ]
        if standard_rows:
            standard_rows.sort(key=lambda row: int(row["k"]))
            final_row = standard_rows[-1]
            ax.plot(
                [0],
                [float(final_row["avg_cost_tokens"])],
                label=method_display_name("standard_generation_independent"),
                **method_style("standard_generation_independent"),
            )
            ax.lines[-1].set_linestyle("None")
    for method in ("baseline_independent", "shared_trace_branch_after_prefix"):
        xs = []
        ys = []
        for reasoning_prefix_tokens, prefix_summary in sorted(prefix_summaries.items()):
            rows = [
                row
                for row in prefix_summary.get("clean_summary_rows", [])
                if row.get("method") == method
                and row.get("pass_at_k") is not None
                and row.get("avg_cost_tokens") is not None
            ]
            if not rows:
                continue
            rows.sort(key=lambda row: int(row["k"]))
            final_row = rows[-1]
            xs.append(int(reasoning_prefix_tokens))
            ys.append(float(final_row["avg_cost_tokens"]))
        if xs:
            ax.plot(xs, ys, label=method_display_name(method), **method_style(method))
    ax.set_xlabel("Reasoning Budget (tokens)")
    ax.set_ylabel("Average Total Generated Tokens Per Prompt")
    ax.set_title("Reasoning Budget vs Required Total Generation")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    plt.tight_layout()
    plt.savefig(reasoning_dir / "reasoning_budget_vs_required_total_generation_styled.png", dpi=220)
    plt.close()

    plt.figure(figsize=(13.5, 8))
    ax = plt.gca()
    for series_name, rows in sorted(grouped_passk.items()):
        rows = [row for row in rows if row.get("pass_at_k") is not None]
        rows.sort(key=lambda row: int(row["k"]))
        if not rows:
            continue
        exemplar = rows[0]
        xs = [int(row["k"]) for row in rows]
        ys = [float(row["pass_at_k"]) for row in rows]
        ax.plot(
            xs,
            ys,
            label=series_name,
            alpha=0.95,
            **family_series_style(str(exemplar["method"]), int(exemplar.get("reasoning_prefix_tokens", 0))),
        )
        for row, x, y in zip(rows, xs, ys):
            ax.annotate(
                f"k={int(row['k'])}",
                (x, y),
                textcoords="offset points",
                xytext=(4, 6),
                fontsize=8,
                alpha=0.9,
            )
    ax.set_xscale("log", base=2)
    ax.set_xticks(pass_at_ks)
    ax.set_xticklabels([str(k) for k in pass_at_ks])
    ax.set_xlabel("k")
    ax.set_ylabel("pass@k")
    ax.set_title("Pass@k Overlay by Reasoning-Budget Color Family")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(passk_dir / "passk_overlay_by_budget_family_colors.png", dpi=220)
    plt.close()

    plt.figure(figsize=(13.5, 8))
    ax = plt.gca()
    for series_name, rows in sorted(grouped_cost.items()):
        rows = [
            row for row in rows if row.get("pass_at_k") is not None and row.get("avg_cost_tokens") is not None
        ]
        rows.sort(key=lambda row: int(row["k"]))
        if not rows:
            continue
        exemplar = rows[0]
        xs = [float(row["avg_cost_tokens"]) for row in rows]
        ys = [float(row["pass_at_k"]) for row in rows]
        ax.plot(
            xs,
            ys,
            label=series_name,
            alpha=0.95,
            **family_series_style(str(exemplar["method"]), int(exemplar.get("reasoning_prefix_tokens", 0))),
        )
        for row, x, y in zip(rows, xs, ys):
            ax.annotate(
                f"k={int(row['k'])}",
                (x, y),
                textcoords="offset points",
                xytext=(4, 6),
                fontsize=8,
                alpha=0.9,
            )
    ax.set_xlabel("Average Generated Tokens Per Prompt")
    ax.set_ylabel("pass@k")
    ax.set_title("Cost vs pass@k by Reasoning-Budget Color Family")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(cost_dir / "cost_vs_passk_overall_family_colors.png", dpi=220)
    plt.close()

    plt.figure(figsize=(11, 7))
    ax = plt.gca()
    if standard_summary is not None:
        standard_rows = [
            row
            for row in standard_summary.get("clean_summary_rows", [])
            if row.get("pass_at_k") is not None and row.get("avg_cost_tokens") is not None
        ]
        if standard_rows:
            standard_rows.sort(key=lambda row: int(row["k"]))
            final_row = standard_rows[-1]
            ax.plot(
                [0],
                [float(final_row["avg_cost_tokens"])],
                label=method_display_name("standard_generation_independent"),
                **method_style("standard_generation_independent"),
            )
            ax.lines[-1].set_linestyle("None")
    for method in ("baseline_independent", "shared_trace_branch_after_prefix"):
        xs = []
        ys = []
        for reasoning_prefix_tokens, prefix_summary in sorted(prefix_summaries.items()):
            rows = [
                row
                for row in prefix_summary.get("clean_summary_rows", [])
                if row.get("method") == method
                and row.get("pass_at_k") is not None
                and row.get("avg_cost_tokens") is not None
            ]
            if not rows:
                continue
            rows.sort(key=lambda row: int(row["k"]))
            final_row = rows[-1]
            xs.append(int(reasoning_prefix_tokens))
            ys.append(float(final_row["avg_cost_tokens"]))
        if xs:
            ax.plot(
                xs,
                ys,
                label=method_display_name(method),
                alpha=0.95,
                **method_style(method),
            )
    ax.set_xlabel("Reasoning Budget (tokens)")
    ax.set_ylabel("Average Total Generated Tokens Per Prompt")
    ax.set_title("Reasoning Budget vs Required Total Generation")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(reasoning_dir / "reasoning_budget_vs_required_total_generation_clear.png", dpi=220)
    plt.close()

    k16_pass_at_ks = [k for k in pass_at_ks if int(k) <= 16]
    if k16_pass_at_ks:
        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        for series_name, rows in sorted(grouped_passk.items()):
            rows = [row for row in rows if row.get("pass_at_k") is not None and int(row["k"]) <= 16]
            rows.sort(key=lambda row: int(row["k"]))
            if not rows:
                continue
            exemplar = rows[0]
            xs = [int(row["k"]) for row in rows]
            ys = [float(row["pass_at_k"]) for row in rows]
            ax.plot(
                xs,
                ys,
                label=series_name,
                **series_style(str(exemplar["method"]), int(exemplar.get("reasoning_prefix_tokens", 0))),
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(k16_pass_at_ks)
        ax.set_xticklabels([str(k) for k in k16_pass_at_ks])
        ax.set_xlabel("k")
        ax.set_ylabel("pass@k")
        ax.set_title("Pass@k Overlay Across Reasoning Budgets (k<=16)")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(passk_dir / "passk_overlay_by_budget_k16.png", dpi=220)
        plt.close()

        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        for series_name, rows in sorted(legacy_grouped_passk.items()):
            rows = [row for row in rows if row.get("pass_at_k") is not None and int(row["k"]) <= 16]
            rows.sort(key=lambda row: int(row["k"]))
            if not rows:
                continue
            exemplar = rows[0]
            xs = [int(row["k"]) for row in rows]
            ys = [float(row["pass_at_k"]) for row in rows]
            ax.plot(
                xs,
                ys,
                label=series_name,
                **series_style(str(exemplar["method"]), int(exemplar.get("reasoning_prefix_tokens", 0))),
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(k16_pass_at_ks)
        ax.set_xticklabels([str(k) for k in k16_pass_at_ks])
        ax.set_xlabel("k")
        ax.set_ylabel("pass@k")
        ax.set_title("pass@k Overlay (k<=16)")
        ax.grid(True, which="both", linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(passk_dir / "passk_overlay_k16.png", dpi=220)
        plt.close()

        plt.figure(figsize=(12, 7))
        ax = plt.gca()
        for series_name, rows in sorted(grouped_cost.items()):
            rows = [
                row
                for row in rows
                if row.get("pass_at_k") is not None and row.get("avg_cost_tokens") is not None and int(row["k"]) <= 16
            ]
            rows.sort(key=lambda row: int(row["k"]))
            if not rows:
                continue
            exemplar = rows[0]
            xs = [float(row["avg_cost_tokens"]) for row in rows]
            ys = [float(row["pass_at_k"]) for row in rows]
            ax.plot(
                xs,
                ys,
                label=series_name,
                **series_style(str(exemplar["method"]), int(exemplar.get("reasoning_prefix_tokens", 0))),
            )
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
        ax.set_title("Cost vs pass@k Overlay (k<=16)")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(cost_dir / "cost_vs_passk_overall_k16.png", dpi=220)
        plt.close()

        plt.figure(figsize=(13.5, 8))
        ax = plt.gca()
        for series_name, rows in sorted(grouped_passk.items()):
            rows = [row for row in rows if row.get("pass_at_k") is not None and int(row["k"]) <= 16]
            rows.sort(key=lambda row: int(row["k"]))
            if not rows:
                continue
            exemplar = rows[0]
            xs = [int(row["k"]) for row in rows]
            ys = [float(row["pass_at_k"]) for row in rows]
            ax.plot(
                xs,
                ys,
                label=series_name,
                alpha=0.95,
                **family_series_style(str(exemplar["method"]), int(exemplar.get("reasoning_prefix_tokens", 0))),
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(k16_pass_at_ks)
        ax.set_xticklabels([str(k) for k in k16_pass_at_ks])
        ax.set_xlabel("k")
        ax.set_ylabel("pass@k")
        ax.set_title("Pass@k Overlay by Reasoning-Budget Color Family (k<=16)")
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(passk_dir / "passk_overlay_by_budget_family_colors_k16.png", dpi=220)
        plt.close()

        plt.figure(figsize=(13.5, 8))
        ax = plt.gca()
        for series_name, rows in sorted(grouped_cost.items()):
            rows = [
                row
                for row in rows
                if row.get("pass_at_k") is not None and row.get("avg_cost_tokens") is not None and int(row["k"]) <= 16
            ]
            rows.sort(key=lambda row: int(row["k"]))
            if not rows:
                continue
            exemplar = rows[0]
            xs = [float(row["avg_cost_tokens"]) for row in rows]
            ys = [float(row["pass_at_k"]) for row in rows]
            ax.plot(
                xs,
                ys,
                label=series_name,
                alpha=0.95,
                **family_series_style(str(exemplar["method"]), int(exemplar.get("reasoning_prefix_tokens", 0))),
            )
        ax.set_xlabel("Average Generated Tokens Per Prompt")
        ax.set_ylabel("pass@k")
        ax.set_title("Cost vs pass@k by Reasoning-Budget Color Family (k<=16)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(cost_dir / "cost_vs_passk_overall_family_colors_k16.png", dpi=220)
        plt.close()

        plt.figure(figsize=(11, 7))
        ax = plt.gca()
        if standard_summary is not None:
            standard_rows = [
                row
                for row in standard_summary.get("clean_summary_rows", [])
                if row.get("pass_at_k") is not None and row.get("avg_cost_tokens") is not None and int(row["k"]) <= 16
            ]
            if standard_rows:
                standard_rows.sort(key=lambda row: int(row["k"]))
                final_row = standard_rows[-1]
                ax.plot(
                    [0],
                    [float(final_row["avg_cost_tokens"])],
                    label=method_display_name("standard_generation_independent"),
                    **method_style("standard_generation_independent"),
                )
                ax.lines[-1].set_linestyle("None")
        for method in ("baseline_independent", "shared_trace_branch_after_prefix"):
            xs = []
            ys = []
            for reasoning_prefix_tokens, prefix_summary in sorted(prefix_summaries.items()):
                rows = [
                    row
                    for row in prefix_summary.get("clean_summary_rows", [])
                    if row.get("method") == method
                    and row.get("pass_at_k") is not None
                    and row.get("avg_cost_tokens") is not None
                    and int(row["k"]) <= 16
                ]
                if not rows:
                    continue
                rows.sort(key=lambda row: int(row["k"]))
                final_row = rows[-1]
                xs.append(int(reasoning_prefix_tokens))
                ys.append(float(final_row["avg_cost_tokens"]))
            if xs:
                ax.plot(xs, ys, label=method_display_name(method), **method_style(method))
        ax.set_xlabel("Reasoning Budget (tokens)")
        ax.set_ylabel("Average Total Generated Tokens Per Prompt")
        ax.set_title("Reasoning Budget vs Required Total Generation (k<=16)")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(reasoning_dir / "reasoning_budget_vs_required_total_generation_k16.png", dpi=220)
        plt.close()


def write_requested_passk_plots(
    root_output_dir: Path,
    prefix_summaries: Dict[int, Dict[str, Any]],
    standard_summary: Optional[Dict[str, Any]],
    pass_at_ks: List[int],
    max_k: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    rows: List[Dict[str, Any]] = []
    for reasoning_prefix_tokens, prefix_summary in sorted(prefix_summaries.items()):
        for row in prefix_summary.get("clean_summary_rows", []):
            if int(row.get("k", 0)) <= max_k:
                row_copy = dict(row)
                row_copy["reasoning_prefix_tokens"] = int(reasoning_prefix_tokens)
                rows.append(row_copy)
    if standard_summary is not None:
        for row in standard_summary.get("clean_summary_rows", []):
            if int(row.get("k", 0)) <= max_k:
                row_copy = dict(row)
                row_copy["reasoning_prefix_tokens"] = 0
                rows.append(row_copy)
    rows = [
        row
        for row in rows
        if row.get("pass_at_k") is not None and row.get("avg_cost_tokens") is not None
    ]
    if not rows:
        return

    requested_dir = root_output_dir / "requested_plots"
    requested_dir.mkdir(parents=True, exist_ok=True)
    budget_markers = {0: "^", 256: "o", 512: "s", 1024: "D", 2048: "P", 4096: "X", 6144: "v"}
    method_linestyles = {
        "baseline_independent": "-",
        "shared_trace_branch_after_prefix": "--",
        "standard_generation_independent": "-.",
    }

    def iter_series() -> Iterable[Tuple[str, int, List[Dict[str, Any]]]]:
        for method in sorted({str(row["method"]) for row in rows}):
            method_rows = [row for row in rows if str(row["method"]) == method]
            for reasoning_prefix_tokens in sorted(
                {int(row["reasoning_prefix_tokens"]) for row in method_rows}
            ):
                series_rows = [
                    row
                    for row in method_rows
                    if int(row["reasoning_prefix_tokens"]) == reasoning_prefix_tokens
                ]
                series_rows.sort(key=lambda row: int(row["k"]))
                yield method, reasoning_prefix_tokens, series_rows

    def label_for(method: str, reasoning_prefix_tokens: int) -> str:
        if reasoning_prefix_tokens == 0:
            return method_display_name(method)
        return f"{method_display_name(method)} / {reasoning_prefix_tokens}"

    def add_series(ax: Any, x_key: str, y_key: str, *, annotate_k: bool = False) -> None:
        for method, reasoning_prefix_tokens, series_rows in iter_series():
            ax.plot(
                [float(row[x_key]) for row in series_rows],
                [float(row[y_key]) for row in series_rows],
                color=technique_color(method),
                marker=budget_markers.get(reasoning_prefix_tokens, "o"),
                linestyle=method_linestyles.get(method, "-"),
                linewidth=2.0,
                markersize=7,
                alpha=0.92,
                label=label_for(method, reasoning_prefix_tokens),
            )
            if annotate_k:
                for row in series_rows:
                    ax.annotate(
                        f"k={int(row['k'])}",
                        (float(row[x_key]), float(row[y_key])),
                        textcoords="offset points",
                        xytext=(4, 5),
                        fontsize=7,
                    )

    plot_specs = [
        ("k_vs_passk_performance.png", "k", "pass_at_k", "k", "Pass@k Performance", "k vs Pass@k Performance", True, False),
        ("memory_usage_vs_passk_performance.png", "avg_cost_tokens", "pass_at_k", "Memory Usage (tokens)", "Pass@k Performance", "Memory Usage vs Pass@k Performance", False, True),
        ("k_vs_generated_tokens.png", "k", "avg_cost_tokens", "k", "Generated Tokens", "k vs Generated Tokens", True, False),
    ]
    for filename, x_key, y_key, xlabel, ylabel, title, log_x, annotate_k in plot_specs:
        plt.figure(figsize=(13, 8))
        ax = plt.gca()
        add_series(ax, x_key, y_key, annotate_k=annotate_k)
        if log_x:
            ax.set_xscale("log", base=2)
            ax.set_xticks(pass_at_ks)
            ax.set_xticklabels([str(k) for k in pass_at_ks])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if y_key == "pass_at_k":
            ax.set_ylim(0.0, 1.0)
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        plt.tight_layout()
        plt.savefig(requested_dir / filename, dpi=240)
        plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    add_series(axes[0], "k", "pass_at_k")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(pass_at_ks)
    axes[0].set_xticklabels([str(k) for k in pass_at_ks])
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Pass@k Performance")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("k vs Pass@k")
    add_series(axes[1], "avg_cost_tokens", "pass_at_k")
    axes[1].set_xlabel("Memory Usage (tokens)")
    axes[1].set_ylabel("Pass@k Performance")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Memory vs Pass@k")
    add_series(axes[2], "k", "avg_cost_tokens")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(pass_at_ks)
    axes[2].set_xticklabels([str(k) for k in pass_at_ks])
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Generated Tokens")
    axes[2].set_title("k vs Generated Tokens")
    for ax in axes:
        ax.grid(True, which="both", linestyle="--", alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", fontsize=8, ncol=4)
    fig.suptitle("Combined Pass@k, Memory, and Token Usage", y=0.99)
    plt.tight_layout(rect=(0, 0, 1, 0.88))
    plt.savefig(requested_dir / "combined_passk_memory_tokens.png", dpi=240)
    plt.close(fig)

    averaged_rows: List[Dict[str, Any]] = []
    for method in ("baseline_independent", "shared_trace_branch_after_prefix"):
        for k in pass_at_ks:
            bucket = [
                row
                for row in rows
                if str(row["method"]) == method and int(row["k"]) == int(k)
            ]
            if bucket:
                averaged_rows.append(
                    {
                        "method": method,
                        "k": k,
                        "pass_at_k": sum(float(row["pass_at_k"]) for row in bucket) / len(bucket),
                        "avg_cost_tokens": sum(float(row["avg_cost_tokens"]) for row in bucket) / len(bucket),
                    }
                )
    if averaged_rows:
        plt.figure(figsize=(10.5, 6.5))
        ax = plt.gca()
        for method in sorted({str(row["method"]) for row in averaged_rows}):
            method_rows = [row for row in averaged_rows if str(row["method"]) == method]
            method_rows.sort(key=lambda row: int(row["k"]))
            ax.plot(
                [float(row["avg_cost_tokens"]) for row in method_rows],
                [float(row["pass_at_k"]) for row in method_rows],
                color=technique_color(method),
                marker="o" if method == "baseline_independent" else "s",
                linestyle=method_linestyles.get(method, "-"),
                linewidth=2.4,
                label=method_display_name(method),
            )
            for row in method_rows:
                ax.annotate(
                    f"k={int(row['k'])}",
                    (float(row["avg_cost_tokens"]), float(row["pass_at_k"])),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )
        ax.set_xlabel("Memory Usage (tokens)")
        ax.set_ylabel("Pass@k Performance")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Average Across Reasoning Budgets")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(requested_dir / "average_across_budgets_memory_vs_passk.png", dpi=240)
        plt.close()

    budget_average_rows: List[Dict[str, Any]] = []
    for method in ("baseline_independent", "shared_trace_branch_after_prefix"):
        for reasoning_prefix_tokens in sorted(
            {
                int(row["reasoning_prefix_tokens"])
                for row in rows
                if str(row["method"]) == method
            }
        ):
            bucket = [
                row
                for row in rows
                if str(row["method"]) == method
                and int(row["reasoning_prefix_tokens"]) == reasoning_prefix_tokens
            ]
            if bucket:
                budget_average_rows.append(
                    {
                        "method": method,
                        "reasoning_prefix_tokens": reasoning_prefix_tokens,
                        "pass_at_k": sum(float(row["pass_at_k"]) for row in bucket) / len(bucket),
                        "avg_cost_tokens": sum(float(row["avg_cost_tokens"]) for row in bucket) / len(bucket),
                    }
                )
    if budget_average_rows:
        plt.figure(figsize=(10.5, 6.5))
        ax = plt.gca()
        for method in sorted({str(row["method"]) for row in budget_average_rows}):
            method_rows = [row for row in budget_average_rows if str(row["method"]) == method]
            method_rows.sort(key=lambda row: int(row["reasoning_prefix_tokens"]))
            ax.plot(
                [float(row["avg_cost_tokens"]) for row in method_rows],
                [float(row["pass_at_k"]) for row in method_rows],
                color=technique_color(method),
                marker="o" if method == "baseline_independent" else "s",
                linestyle=method_linestyles.get(method, "-"),
                linewidth=2.4,
                label=method_display_name(method),
            )
            for row in method_rows:
                ax.annotate(
                    str(int(row["reasoning_prefix_tokens"])),
                    (float(row["avg_cost_tokens"]), float(row["pass_at_k"])),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )
        ax.set_xlabel("Memory Usage (tokens)")
        ax.set_ylabel("Pass@k Performance")
        ax.set_ylim(0.0, 1.0)
        ax.set_title("Average Across Pass Values by Reasoning Budget")
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.savefig(requested_dir / "average_across_passes_by_budget.png", dpi=240)
        plt.close()


def build_manifest_payload(
    args: argparse.Namespace,
    runtime_config: RuntimeConfig,
    scheduler_config: Optional[SchedulerConfig],
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
    runtime_probe_metadata: Optional[Dict[str, Any]] = None,
    scheduler_metadata: Optional[Dict[str, Any]] = None,
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
        "experiment_mode": args.experiment_mode,
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
        "throughput_profile": args.throughput_profile,
        "visible_gpu_count": runtime_config.visible_gpu_count,
        "per_gpu_memory_gb": runtime_config.per_gpu_memory_gb,
        "max_new_tokens": args.max_new_tokens,
        "reasoning_prefix_token_values": reasoning_prefix_token_values
        or parse_reasoning_prefix_token_values(args.reasoning_prefix_token_values),
        "checkpoint_matched_prompts_step": args.checkpoint_matched_prompts_step,
        "compact_jsonl": bool(getattr(args, "compact_jsonl", False)),
        "examples": len(examples),
        "selected_source_indices": [
            int((example.metadata or {}).get("source_index", example.prompt_index))
            for example in examples
        ],
        "git_commit": git_commit,
        "runtime_config": runtime_config_payload(runtime_config),
        "server_args": serialize_server_args(server_args),
        "assistant_prefill": prompt_build_info.assistant_prefill,
        "prompt_build_mode": prompt_build_info.prompt_build_mode,
        "sampling_defaults": base_sampling_params(
            args.max_new_tokens,
            enable_soft_thinking=True,
            sampling_overrides=current_sampling_overrides(args),
        ),
        "fixed_prefix_sampling_defaults": {
            str(prefix_tokens): fixed_prefix_sampling_params(
                prefix_tokens,
                sampling_overrides=current_sampling_overrides(args),
            )
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
            ),
            sampling_overrides=current_sampling_overrides(args),
        ),
        "standard_generation_sampling_defaults": standard_generation_sampling_params(
            args.max_new_tokens,
            sampling_overrides=current_sampling_overrides(args),
        ),
        "sampling_overrides": current_sampling_overrides(args),
        "run_status": run_status,
    }
    if scheduler_config is not None:
        manifest["scheduler_config"] = asdict(scheduler_config)
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
    if runtime_probe_metadata is not None:
        manifest["runtime_probe"] = runtime_probe_metadata
    if scheduler_metadata is not None:
        manifest["scheduler_metadata"] = scheduler_metadata
    return manifest


def artifact_stem(base_name: str, artifact_label: Optional[str] = None) -> str:
    if artifact_label:
        return f"{base_name}_{artifact_label}"
    return base_name


def write_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    runtime_config: RuntimeConfig,
    scheduler_config: Optional[SchedulerConfig],
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
    runtime_probe_metadata: Optional[Dict[str, Any]] = None,
    scheduler_metadata: Optional[Dict[str, Any]] = None,
) -> None:
    manifest = build_manifest_payload(
        args,
        runtime_config,
        scheduler_config,
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
        runtime_probe_metadata=runtime_probe_metadata,
        scheduler_metadata=scheduler_metadata,
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
    write_requested_passk_plots(
        root_output_dir,
        prefix_summaries,
        standard_summary,
        pass_at_ks,
        max_k,
    )


def write_prefix_sweep_summary_locked(
    root_output_dir: Path,
    prefix_summaries: Dict[int, Dict[str, Any]],
    standard_summary: Optional[Dict[str, Any]],
    pass_at_ks: List[int],
    max_k: int,
    run_timing: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[int, Dict[str, Any]], Optional[Dict[str, Any]]]:
    try:
        import fcntl
    except ModuleNotFoundError:
        write_prefix_sweep_summary(
            root_output_dir,
            prefix_summaries,
            standard_summary,
            pass_at_ks,
            max_k,
            run_timing=run_timing,
        )
        return prefix_summaries, standard_summary

    root_output_dir.mkdir(parents=True, exist_ok=True)
    lock_path = root_output_dir / ".summary.lock"
    with lock_path.open("w", encoding="utf-8") as lock_fp:
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
        loaded_prefix_summaries, loaded_standard_summary = load_existing_sweep_summaries(
            root_output_dir,
            pass_at_ks=pass_at_ks,
            max_k=max_k,
        )
        merged_prefix_summaries = dict(loaded_prefix_summaries)
        merged_prefix_summaries.update(prefix_summaries)
        merged_standard_summary = standard_summary or loaded_standard_summary
        write_prefix_sweep_summary(
            root_output_dir,
            merged_prefix_summaries,
            merged_standard_summary,
            pass_at_ks,
            max_k,
            run_timing=run_timing,
        )
        fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
    return merged_prefix_summaries, merged_standard_summary


def run_fixed_prefix_experiment(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    runtime_config: RuntimeConfig,
    scheduler_config: SchedulerConfig,
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
    events_path = output_dir / "events.jsonl"
    event_logger = StructuredEventLogger(events_path)
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
        scheduler_config,
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
    task_items: List[Dict[str, Any]] = []
    for example in examples:
        run_baseline = "baseline_independent" in methods and not completed_pairs.get(
            ("baseline_independent", example.prompt_index),
            False,
        )
        run_shared_trace = "shared_trace_branch_after_prefix" in methods and not completed_pairs.get(
            ("shared_trace_branch_after_prefix", example.prompt_index),
            False,
        )
        if run_baseline or run_shared_trace:
            task_items.append(
                {
                    "example": example,
                    "run_baseline": run_baseline,
                    "run_shared_trace": run_shared_trace,
                    "existing_baseline_sample_indices": usable_sample_indices_for_prompt(
                        sample_rows,
                        method="baseline_independent",
                        prompt_index=example.prompt_index,
                    ),
                }
            )

    def submit_task(task_item: Dict[str, Any], target_dp_rank: int) -> FixedPrefixPromptTaskResult:
        return run_fixed_prefix_prompt_task(
            args=args,
            benchmark=args.benchmark,
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            timeout=args.server_timeout_seconds,
            example=task_item["example"],
            target_dp_rank=target_dp_rank,
            logger=logger,
            run_baseline=bool(task_item["run_baseline"]),
            run_shared_trace=bool(task_item["run_shared_trace"]),
            existing_baseline_sample_indices=set(
                task_item.get("existing_baseline_sample_indices") or set()
            ),
        )

    def handle_result(result: FixedPrefixPromptTaskResult) -> None:
        if result.event_records:
            append_records(events_path, result.event_records)
        if result.baseline_attempts:
            serialized = [asdict(record) for record in result.baseline_attempts]
            append_records(attempts_path, serialized)
            attempt_rows.extend(serialized)
        if result.shared_attempts:
            serialized = [asdict(record) for record in result.shared_attempts]
            append_records(attempts_path, serialized)
            attempt_rows.extend(serialized)
        if result.baseline_result is not None:
            serialized_result = asdict(result.baseline_result)
            append_records(baseline_prompts_path, [serialized_result])
            baseline_prompt_rows.append(serialized_result)
            completed_pairs[("baseline_independent", result.prompt_index)] = True
        if result.baseline_records:
            serialized_records = [asdict(record) for record in result.baseline_records]
            append_records(samples_path, serialized_records)
            sample_rows.extend(serialized_records)
        if result.parent_trace is not None:
            serialized_parent = asdict(result.parent_trace)
            append_records(parents_path, [serialized_parent])
            parent_rows.append(serialized_parent)
            completed_pairs[("shared_trace_branch_after_prefix", result.prompt_index)] = True
        if result.child_records:
            serialized_child_records = [asdict(record) for record in result.child_records]
            append_records(samples_path, serialized_child_records)
            sample_rows.extend(serialized_child_records)
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

    scheduler_metadata = run_prompt_tasks_concurrently(
        task_items=task_items,
        scheduler_config=scheduler_config,
        effective_dp_size=runtime_config.effective_dp_size,
        submit_task=submit_task,
        handle_result=handle_result,
        logger=logger,
        event_logger=event_logger,
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
        scheduler_config,
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
        scheduler_metadata=scheduler_metadata,
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
    scheduler_config: SchedulerConfig,
    pass_at_ks: List[int],
    examples: List[Example],
    prompt_build_info: PromptBuildInfo,
    client: SGLangRestClient,
) -> Dict[str, Any]:
    run_started_at_unix = time.time()
    logger = configure_logging(output_dir, logger_name="compare_passk_aime.standard_generation")
    events_path = output_dir / "events.jsonl"
    event_logger = StructuredEventLogger(events_path)
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
        scheduler_config,
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
    task_items = [
        {"example": example}
        for example in examples
        if not completed_pairs.get(("standard_generation_independent", example.prompt_index), False)
    ]

    def submit_task(task_item: Dict[str, Any], target_dp_rank: int) -> StandardPromptTaskResult:
        return run_standard_generation_prompt_task(
            args=args,
            benchmark=args.benchmark,
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            timeout=args.server_timeout_seconds,
            example=task_item["example"],
            target_dp_rank=target_dp_rank,
            logger=logger,
        )

    def handle_result(result: StandardPromptTaskResult) -> None:
        if result.event_records:
            append_records(events_path, result.event_records)
        if result.attempt_records:
            serialized_attempts = [asdict(record) for record in result.attempt_records]
            append_records(attempts_path, serialized_attempts)
            attempt_rows.extend(serialized_attempts)
        if result.prompt_result is not None:
            serialized_result = asdict(result.prompt_result)
            append_records(prompt_rows_path, [serialized_result])
            prompt_rows.append(serialized_result)
            completed_pairs[("standard_generation_independent", result.prompt_index)] = True
        if result.sample_records:
            serialized_records = [asdict(record) for record in result.sample_records]
            append_records(samples_path, serialized_records)
            sample_rows.extend(serialized_records)
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

    scheduler_metadata = run_prompt_tasks_concurrently(
        task_items=task_items,
        scheduler_config=scheduler_config,
        effective_dp_size=runtime_config.effective_dp_size,
        submit_task=submit_task,
        handle_result=handle_result,
        logger=logger,
        event_logger=event_logger,
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
        scheduler_config,
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
        scheduler_metadata=scheduler_metadata,
    )
    logger.info(
        "[setup] Standard generation finished in %s (%.1fs). Summary written to %s.",
        run_timing.get("wall_clock_hms"),
        float(run_timing.get("wall_clock_seconds", 0.0)),
        output_dir / "summary.md",
    )
    return summary_json


def shared_group_method_name(group_size: int) -> str:
    return f"shared_trace_group_{int(group_size)}"


def load_existing_branch_ablation_group_summaries(
    root_output_dir: Path,
    *,
    pass_at_ks: List[int],
    max_k: int,
) -> Dict[int, Dict[str, Any]]:
    group_summaries: Dict[int, Dict[str, Any]] = {}
    for group_dir in sorted(root_output_dir.glob("shared_every_*")):
        if not group_dir.is_dir():
            continue
        try:
            group_size = int(group_dir.name.rsplit("_", 1)[1])
        except ValueError:
            continue
        summary = summarize_existing_output_dir(
            group_dir,
            pass_at_ks=pass_at_ks,
            max_k=max_k,
            includes_shared_trace=False,
        )
        if summary is not None:
            group_summaries[group_size] = summary
    return group_summaries


def safe_float_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def run_shared_grouped_trace_for_prompt(
    *,
    args: argparse.Namespace,
    benchmark: str,
    client: SGLangRestClient,
    example: Example,
    target_dp_rank: int,
    logger: logging.Logger,
    event_logger: StructuredEventLogger,
    group_size: int,
    existing_sample_indices: Optional[set[int]] = None,
) -> Tuple[BaselinePromptResult, List[SampleRecord]]:
    reasoning_prefix_tokens = int(args.current_reasoning_prefix_tokens)
    decode_budget = continuation_token_budget(args.max_new_tokens, reasoning_prefix_tokens)
    method = shared_group_method_name(group_size)
    existing_sample_indices = existing_sample_indices or set()
    sampling_overrides = current_sampling_overrides(args)
    parent_params = fixed_prefix_sampling_params(
        reasoning_prefix_tokens,
        sampling_overrides=sampling_overrides,
    )
    child_params = child_sampling_params(
        decode_budget,
        sampling_overrides=sampling_overrides,
    )
    child_seeds = planned_child_seeds(args.seed, example.prompt_index, args.max_k)
    slot_statuses: List[Dict[str, Any]] = [
        {
            "slot_index": slot_index,
            "branch_group_index": slot_index // group_size,
            "prefix_completion_tokens": 0,
            "forced_think_end_tokens": 0,
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
    for slot_index in existing_sample_indices:
        if 0 <= slot_index < len(slot_statuses):
            slot_statuses[slot_index]["accepted_for_eval"] = True
    records: List[SampleRecord] = []
    logger.info(
        "[branch-ablation] Prompt %s: repairing shared traces every %s samples at prefix=%s with %s/%s existing samples.",
        example.prompt_index,
        group_size,
        reasoning_prefix_tokens,
        len(existing_sample_indices),
        args.max_k,
    )
    event_logger.log(
        "branch_ablation_prompt_start",
        method=method,
        prompt_index=example.prompt_index,
        target_dp_rank=target_dp_rank,
        group_size=group_size,
        reasoning_prefix_tokens=reasoning_prefix_tokens,
        required_sample_count=args.max_k,
        existing_usable_sample_count=len(existing_sample_indices),
    )

    for group_index, start_index in enumerate(range(0, args.max_k, group_size)):
        end_index = min(start_index + group_size, args.max_k)
        group_count = end_index - start_index
        group_sample_indices = set(range(start_index, end_index))
        if group_sample_indices.issubset(existing_sample_indices):
            event_logger.log(
                "branch_ablation_group_repair_skipped",
                method=method,
                prompt_index=example.prompt_index,
                group_index=group_index,
                start_index=start_index,
                end_index=end_index,
                reason="group_already_has_full_usable_samples",
            )
            continue
        session_id = (
            f"aime-shared-g{group_size}-p{example.prompt_index}-"
            f"t{reasoning_prefix_tokens}-group{group_index}"
        )
        parent_rid = (
            f"shared-g{group_size}-p{example.prompt_index}-"
            f"t{reasoning_prefix_tokens}-group{group_index}-prefix"
        )
        child_rids = [
            f"{parent_rid}-child-{sample_index}"
            for sample_index in range(start_index, end_index)
        ]
        session_opened = False
        try:
            client.open_session(args.capacity_of_str_len, session_id=session_id)
            session_opened = True
            parent_output = client.generate(
                input_ids=example.prompt_ids,
                sampling_params=parent_params,
                rid=parent_rid,
                data_parallel_rank=target_dp_rank,
                session_params={"id": session_id},
            )
            if not isinstance(parent_output, dict):
                raise RuntimeError(f"Unexpected branch-ablation parent output: {parent_output!r}")
            parent_meta = parent_output.get("meta_info", {})
            parent_completion_tokens = int(parent_meta.get("completion_tokens", 0))
            parent_latency_seconds = float(parent_meta.get("e2e_latency", 0.0))
            parent_finish_reason = parent_meta.get("finish_reason")
            parent_text = reconstruct_assistant_text(
                example.assistant_prefill,
                parent_output.get("text", ""),
            )
            prefix_reached = reached_reasoning_prefix(parent_output, reasoning_prefix_tokens)
            for slot_index in range(start_index, end_index):
                slot_statuses[slot_index]["prefix_completion_tokens"] = parent_completion_tokens
                slot_statuses[slot_index]["prefix_latency_seconds"] = parent_latency_seconds
                slot_statuses[slot_index]["prefix_finish_reason"] = parent_finish_reason
                slot_statuses[slot_index]["prefix_reached"] = prefix_reached
            event_logger.log(
                "branch_ablation_parent_done",
                method=method,
                prompt_index=example.prompt_index,
                group_index=group_index,
                target_dp_rank=target_dp_rank,
                reasoning_prefix_tokens=reasoning_prefix_tokens,
                prefix_reached=prefix_reached,
                completion_tokens=parent_completion_tokens,
                latency_seconds=parent_latency_seconds,
                finish_reason=parent_finish_reason,
            )
            if not prefix_reached:
                for slot_index in range(start_index, end_index):
                    slot_statuses[slot_index]["failure_reason"] = "prefix_not_reached"
                continue

            fork_info = client.fork_request(
                session_id=session_id,
                parent_rid=parent_rid,
                child_count=group_count,
                child_rids=child_rids,
                child_seeds=child_seeds[start_index:end_index],
                target_dp_rank=target_dp_rank,
                allow_non_eot_branch=True,
                force_think_end=True,
            )
            if not fork_info.get("success", False):
                for slot_index in range(start_index, end_index):
                    slot_statuses[slot_index]["failure_reason"] = "fork_failed"
                event_logger.log(
                    "branch_ablation_fork_failed",
                    method=method,
                    prompt_index=example.prompt_index,
                    group_index=group_index,
                    target_dp_rank=target_dp_rank,
                    message=fork_info.get("message", "fork_request failed"),
                )
                continue
            forced_think_end_tokens = int(
                fork_info.get("forced_think_end_token_count", 0)
            )
            branch_parent_text = (
                parent_text + THINK_END_TAG
                if forced_think_end_tokens and THINK_END_TAG not in parent_text
                else parent_text
            )
            event_logger.log(
                "branch_ablation_fork_done",
                method=method,
                prompt_index=example.prompt_index,
                group_index=group_index,
                target_dp_rank=target_dp_rank,
                group_size=group_count,
                cacheable_token_count=fork_info.get("cacheable_token_count", 0),
                forced_think_end_tokens=forced_think_end_tokens,
            )
            outputs = normalize_generate_outputs(
                client.generate(
                    input_ids=[fork_info["branch_input_ids"]] * group_count,
                    sampling_params=[dict(child_params) for _ in range(group_count)],
                    rid=child_rids,
                    data_parallel_rank=target_dp_rank,
                ),
                group_count,
                "branch-ablation children batch",
            )
            for offset, output in enumerate(outputs):
                sample_index = start_index + offset
                continuation_meta = output.get("meta_info", {})
                prefix_cost_for_sample = (
                    parent_completion_tokens + forced_think_end_tokens
                    if offset == 0
                    else 0
                )
                forced_cost_for_sample = forced_think_end_tokens if offset == 0 else 0
                slot_status = slot_statuses[sample_index]
                slot_status["forced_think_end_tokens"] = forced_think_end_tokens
                slot_status["continuation_completion_tokens"] = int(
                    continuation_meta.get("completion_tokens", 0)
                )
                slot_status["continuation_latency_seconds"] = float(
                    continuation_meta.get("e2e_latency", 0.0)
                )
                slot_status["accepted_for_eval"] = True
                record = make_sample_record(
                    benchmark=benchmark,
                    method=method,
                    example=example,
                    sample_index=sample_index,
                    rid=child_rids[offset],
                    target_dp_rank=target_dp_rank,
                    output=output,
                    planned_seed=child_seeds[sample_index],
                    max_new_tokens=decode_budget,
                    accepted_attempt_index=None,
                    prefix_text=branch_parent_text,
                    prefix_completion_tokens=prefix_cost_for_sample,
                    forced_think_end_tokens=forced_cost_for_sample,
                    branch_group_index=group_index,
                    reasoning_prefix_tokens=reasoning_prefix_tokens,
                )
                records.append(record)
                event_logger.log(
                    "branch_ablation_child_sample",
                    method=method,
                    prompt_index=example.prompt_index,
                    sample_index=sample_index,
                    group_index=group_index,
                    target_dp_rank=target_dp_rank,
                    correct=record.correct,
                    score=record.score,
                    prefix_completion_tokens=record.prefix_completion_tokens,
                    completion_tokens=record.completion_tokens,
                    forced_think_end_tokens=record.forced_think_end_tokens,
                    finish_reason=record.finish_reason,
                )
        finally:
            if session_opened:
                client.close_session(session_id)

    records.sort(key=lambda record: int(record.sample_index))
    combined_usable_count = len(
        set(existing_sample_indices)
        | {int(record.sample_index) for record in records}
    )
    success = combined_usable_count >= args.max_k
    total_completion_tokens_spent = sum(
        int(record.prefix_completion_tokens) + int(record.completion_tokens)
        for record in records
    )
    total_latency_seconds_spent = sum(
        float(status["prefix_latency_seconds"]) + float(status["continuation_latency_seconds"])
        for status in slot_statuses
        if status["accepted_for_eval"]
    )
    reject_reason_counts_payload = dict(
        Counter(
            str(status["failure_reason"])
            for status in slot_statuses
            if status["failure_reason"]
        )
    )
    prompt_result = BaselinePromptResult(
        benchmark=benchmark,
        prompt_index=example.prompt_index,
        problem_id=example.problem_id,
        target_dp_rank=target_dp_rank,
        success=success,
        message=(
            f"Completed shared trace every {group_size} generations."
            if success
            else f"Only produced {combined_usable_count}/{args.max_k} grouped shared samples."
        ),
        required_sample_count=args.max_k,
        usable_sample_count=combined_usable_count,
        attempts_used=sum(
            1
            for start_index in range(0, args.max_k, group_size)
            if not set(range(start_index, min(start_index + group_size, args.max_k))).issubset(
                existing_sample_indices
            )
        ),
        total_completion_tokens_spent=total_completion_tokens_spent,
        total_latency_seconds_spent=total_latency_seconds_spent,
        reject_reason_counts=reject_reason_counts_payload,
        slot_statuses=slot_statuses,
        reasoning_prefix_tokens=reasoning_prefix_tokens,
        method=method,
    )
    return prompt_result, records


def run_branch_ablation_group_prompt_task(
    *,
    args: argparse.Namespace,
    benchmark: str,
    host: str,
    port: int,
    api_key: Optional[str],
    timeout: int,
    example: Example,
    target_dp_rank: int,
    logger: logging.Logger,
    group_size: int,
    existing_sample_indices: Optional[set[int]] = None,
) -> StandardPromptTaskResult:
    client = make_rest_client(host=host, port=port, api_key=api_key, timeout=timeout)
    event_logger = BufferedEventLogger()
    try:
        prompt_result, sample_records = run_shared_grouped_trace_for_prompt(
            args=args,
            benchmark=benchmark,
            client=client,
            example=example,
            target_dp_rank=target_dp_rank,
            logger=logger,
            event_logger=event_logger,
            group_size=group_size,
            existing_sample_indices=existing_sample_indices,
        )
        return StandardPromptTaskResult(
            prompt_index=example.prompt_index,
            target_dp_rank=target_dp_rank,
            prompt_result=prompt_result,
            sample_records=sample_records,
            attempt_records=[],
            event_records=list(event_logger.records),
        )
    finally:
        client.close()


def run_branch_ablation_group_experiment(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    runtime_config: RuntimeConfig,
    scheduler_config: SchedulerConfig,
    pass_at_ks: List[int],
    examples: List[Example],
    prompt_build_info: PromptBuildInfo,
    multiplex_self_check: Optional[MultiplexSelfCheckResult],
    reasoning_prefix_tokens: int,
    group_size: int,
) -> Dict[str, Any]:
    run_started_at_unix = time.time()
    logger = configure_logging(
        output_dir,
        logger_name=f"compare_passk_aime.branch_group_{group_size}",
    )
    events_path = output_dir / "events.jsonl"
    event_logger = StructuredEventLogger(events_path)
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

    args.current_reasoning_prefix_tokens = reasoning_prefix_tokens
    method = shared_group_method_name(group_size)
    write_manifest(
        output_dir,
        args,
        runtime_config,
        scheduler_config,
        pass_at_ks,
        examples,
        [method],
        prompt_build_info,
        multiplex_self_check=multiplex_self_check,
        run_status="running",
        run_timing=build_run_timing(run_started_at_unix),
        reasoning_prefix_tokens=reasoning_prefix_tokens,
        reasoning_prefix_token_values=[reasoning_prefix_tokens],
    )

    completed_pairs = determine_completed_prompts(prompt_rows, [])
    task_items = [
        {
            "example": example,
            "existing_sample_indices": usable_sample_indices_for_prompt(
                sample_rows,
                method=method,
                prompt_index=example.prompt_index,
            ),
        }
        for example in examples
        if not completed_pairs.get((method, example.prompt_index), False)
    ]

    def submit_task(task_item: Dict[str, Any], target_dp_rank: int) -> StandardPromptTaskResult:
        return run_branch_ablation_group_prompt_task(
            args=args,
            benchmark=args.benchmark,
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            timeout=args.server_timeout_seconds,
            example=task_item["example"],
            target_dp_rank=target_dp_rank,
            logger=logger,
            group_size=group_size,
            existing_sample_indices=set(task_item.get("existing_sample_indices") or set()),
        )

    def handle_result(result: StandardPromptTaskResult) -> None:
        if result.event_records:
            append_records(events_path, result.event_records)
        if result.prompt_result is not None:
            serialized_result = asdict(result.prompt_result)
            append_records(prompt_rows_path, [serialized_result])
            prompt_rows.append(serialized_result)
            completed_pairs[(method, result.prompt_index)] = True
        if result.sample_records:
            serialized_records = [asdict(record) for record in result.sample_records]
            append_records(samples_path, serialized_records)
            sample_rows.extend(serialized_records)
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

    scheduler_metadata = run_prompt_tasks_concurrently(
        task_items=task_items,
        scheduler_config=scheduler_config,
        effective_dp_size=runtime_config.effective_dp_size,
        submit_task=submit_task,
        handle_result=handle_result,
        logger=logger,
        event_logger=event_logger,
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
        scheduler_config,
        pass_at_ks,
        examples,
        [method],
        prompt_build_info,
        multiplex_self_check=multiplex_self_check,
        summary_json=summary_json,
        run_status="completed",
        run_timing=run_timing,
        reasoning_prefix_tokens=reasoning_prefix_tokens,
        reasoning_prefix_token_values=[reasoning_prefix_tokens],
        scheduler_metadata=scheduler_metadata,
    )
    return summary_json


def memory_match_topup_method_name(group_size: int) -> str:
    return f"memory_match_shared_every_{int(group_size)}_topup"


def sample_token_cost(row: Dict[str, Any]) -> int:
    return int(row.get("prefix_completion_tokens", 0) or 0) + int(row.get("completion_tokens", 0) or 0)


def value_to_bool_or_none(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    lowered = str(value).strip().lower()
    if lowered in {"true", "1", "yes"}:
        return True
    if lowered in {"false", "0", "no"}:
        return False
    return None


def value_to_float_or_none(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(value_float):
        return None
    return value_float


def mean_std_optional(values: Iterable[Optional[float]]) -> Tuple[Optional[float], Optional[float], int]:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None, None, 0
    if len(clean) == 1:
        return clean[0], 0.0, 1
    return statistics.mean(clean), statistics.stdev(clean), len(clean)


def read_csv_dict_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fp:
        return list(csv.DictReader(fp))


def infer_repeat_index_from_path(path: Path) -> Optional[int]:
    for part in path.resolve().parts:
        match = re.fullmatch(r"repeat_(\d+)", part)
        if match:
            return int(match.group(1))
    return None


def find_condition_dir(root: Path, condition_dir_name: str) -> Path:
    direct = root / condition_dir_name
    if (direct / "summary.json").exists() or (direct / "samples.jsonl").exists():
        return direct
    matches = sorted(
        path
        for path in root.glob(f"**/{condition_dir_name}")
        if path.is_dir() and ((path / "summary.json").exists() or (path / "samples.jsonl").exists())
    )
    if not matches:
        raise FileNotFoundError(f"Could not find condition directory {condition_dir_name!r} under {root}")
    return matches[0]


def find_condition_dirs(root: Path, condition_dir_name: str) -> List[Path]:
    direct = root / condition_dir_name
    matches: List[Path] = []
    if (direct / "summary.json").exists() or (direct / "samples.jsonl").exists():
        matches.append(direct)
    matches.extend(
        path
        for path in root.glob(f"**/{condition_dir_name}")
        if path.is_dir() and ((path / "summary.json").exists() or (path / "samples.jsonl").exists())
    )
    unique: List[Path] = []
    seen: set[Path] = set()
    for path in sorted(matches):
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    if not unique:
        raise FileNotFoundError(f"Could not find condition directory {condition_dir_name!r} under {root}")
    return unique


def load_condition_summary_for_memory_match(
    condition_dir: Path,
    *,
    pass_at_ks: List[int],
    max_k: int,
) -> Dict[str, Any]:
    summary = summarize_existing_output_dir(
        condition_dir,
        pass_at_ks=pass_at_ks,
        max_k=max_k,
        includes_shared_trace=False,
    )
    if summary is None:
        raise FileNotFoundError(f"No usable summary data found in {condition_dir}")
    return summary


def load_condition_summaries_for_memory_match(
    condition_dirs: List[Path],
    *,
    pass_at_ks: List[int],
    max_k: int,
) -> Dict[str, Any]:
    summaries = [
        load_condition_summary_for_memory_match(
            condition_dir,
            pass_at_ks=pass_at_ks,
            max_k=max_k,
        )
        for condition_dir in condition_dirs
    ]
    if len(summaries) == 1:
        return summaries[0]

    merged = dict(summaries[0])
    merged_prompt_rows: List[Dict[str, Any]] = []
    seen_prompt_methods: set[Tuple[int, str]] = set()
    for summary in summaries:
        rows = summary.get("all_per_prompt_rows") or summary.get("per_prompt_rows") or []
        for row in rows:
            key = (int(row.get("prompt_index", -1)), str(row.get("method", "")))
            if key in seen_prompt_methods:
                continue
            seen_prompt_methods.add(key)
            merged_prompt_rows.append(row)
    merged["all_per_prompt_rows"] = merged_prompt_rows
    merged["per_prompt_rows"] = merged_prompt_rows
    merged["merged_condition_dirs"] = [str(path) for path in condition_dirs]
    return merged


def per_prompt_row_lookup(summary: Dict[str, Any], method: str) -> Dict[int, Dict[str, Any]]:
    rows = summary.get("all_per_prompt_rows") or summary.get("per_prompt_rows") or []
    return {
        int(row["prompt_index"]): row
        for row in rows
        if str(row.get("method", "")) == method
    }


def build_memory_match_prompt_row(
    *,
    group_size: int,
    topup_generator: str = "",
    topup_group_size: Optional[int] = None,
    prompt_index: int,
    problem_id: str,
    target_cost_tokens: Optional[float],
    target_correct: Optional[bool],
    before_cost_tokens: Optional[float],
    before_correct: Optional[bool],
    topup_records: List[Dict[str, Any]],
    base_k: int = 32,
    status: str = "completed",
    message: str = "",
) -> Dict[str, Any]:
    cumulative_cost = float(before_cost_tokens or 0.0)
    after_correct = bool(before_correct)
    sorted_records = sorted(topup_records, key=lambda row: int(row.get("sample_index", 0)))
    for record in sorted_records:
        cumulative_cost += sample_token_cost(record)
        after_correct = after_correct or bool(record.get("correct", False))
    target = None if target_cost_tokens is None else float(target_cost_tokens)
    reached_target = target is not None and cumulative_cost >= target
    return {
        "group_size": int(group_size),
        "topup_generator": topup_generator,
        "topup_group_size": None if topup_group_size is None else int(topup_group_size),
        "prompt_index": int(prompt_index),
        "problem_id": problem_id,
        "target_cost_tokens": target,
        "target_correct": None if target_correct is None else bool(target_correct),
        "before_cost_tokens": None if before_cost_tokens is None else float(before_cost_tokens),
        "after_cost_tokens": cumulative_cost,
        "before_correct": None if before_correct is None else bool(before_correct),
        "after_correct": after_correct,
        "topup_count": len(sorted_records),
        "effective_k": int(base_k) + len(sorted_records),
        "reached_target": reached_target,
        "status": status,
        "message": message,
        "completed_at_unix": time.time(),
    }


def generate_memory_match_fixed_sample(
    *,
    args: argparse.Namespace,
    benchmark: str,
    client: SGLangRestClient,
    example: Example,
    target_dp_rank: int,
    logger: logging.Logger,
    event_logger: StructuredEventLogger,
    group_size: int,
    sample_index: int,
) -> Optional[SampleRecord]:
    reasoning_prefix_tokens = int(args.current_reasoning_prefix_tokens)
    decode_budget = continuation_token_budget(args.max_new_tokens, reasoning_prefix_tokens)
    sampling_overrides = current_sampling_overrides(args)
    prefix_params = fixed_prefix_sampling_params(
        reasoning_prefix_tokens,
        sampling_overrides=sampling_overrides,
    )
    continuation_params = child_sampling_params(
        decode_budget,
        sampling_overrides=sampling_overrides,
    )
    method = memory_match_topup_method_name(group_size)
    session_id = (
        f"memory-match-g{group_size}-p{example.prompt_index}-"
        f"t{reasoning_prefix_tokens}-sample{sample_index}"
    )
    prefix_rid = f"{session_id}-prefix"
    continuation_rid = f"{session_id}-continuation"
    planned_seed = args.seed * 100000 + example.prompt_index * 1000 + sample_index
    session_opened = False
    try:
        client.open_session(args.capacity_of_str_len, session_id=session_id)
        session_opened = True
        prefix_output = client.generate(
            input_ids=example.prompt_ids,
            sampling_params=prefix_params,
            rid=prefix_rid,
            data_parallel_rank=target_dp_rank,
            session_params={"id": session_id},
        )
        if not isinstance(prefix_output, dict):
            raise RuntimeError(f"Unexpected memory-match prefix output: {prefix_output!r}")
        prefix_meta = prefix_output.get("meta_info", {})
        prefix_completion_tokens = int(prefix_meta.get("completion_tokens", 0))
        prefix_reached = reached_reasoning_prefix(prefix_output, reasoning_prefix_tokens)
        event_logger.log(
            "memory_match_prefix_sample",
            method=method,
            group_size=group_size,
            prompt_index=example.prompt_index,
            sample_index=sample_index,
            prefix_reached=prefix_reached,
            completion_tokens=prefix_completion_tokens,
            finish_reason=prefix_meta.get("finish_reason"),
        )
        if not prefix_reached:
            logger.warning(
                "[memory-match] Prompt %s group %s top-up sample %s did not reach prefix.",
                example.prompt_index,
                group_size,
                sample_index,
            )
            return None
        prefix_text = reconstruct_assistant_text(
            example.assistant_prefill,
            prefix_output.get("text", ""),
        )
        fork_info = client.fork_request(
            session_id=session_id,
            parent_rid=prefix_rid,
            child_count=1,
            child_rids=[continuation_rid],
            child_seeds=[planned_seed],
            target_dp_rank=target_dp_rank,
            allow_non_eot_branch=True,
            force_think_end=True,
        )
        if not fork_info.get("success", False):
            event_logger.log(
                "memory_match_fork_failed",
                method=method,
                group_size=group_size,
                prompt_index=example.prompt_index,
                sample_index=sample_index,
                message=fork_info.get("message", "fork_request failed"),
            )
            return None
        forced_think_end_tokens = int(fork_info.get("forced_think_end_token_count", 0))
        branch_prefix_text = (
            prefix_text + THINK_END_TAG
            if forced_think_end_tokens and THINK_END_TAG not in prefix_text
            else prefix_text
        )
        outputs = normalize_generate_outputs(
            client.generate(
                input_ids=[fork_info["branch_input_ids"]],
                sampling_params=[dict(continuation_params)],
                rid=[continuation_rid],
                data_parallel_rank=target_dp_rank,
            ),
            1,
            "memory-match fixed top-up",
        )
        record = make_sample_record(
            benchmark=benchmark,
            method=method,
            example=example,
            sample_index=sample_index,
            rid=continuation_rid,
            target_dp_rank=target_dp_rank,
            output=outputs[0],
            planned_seed=planned_seed,
            max_new_tokens=decode_budget,
            accepted_attempt_index=None,
            prefix_text=branch_prefix_text,
            prefix_completion_tokens=prefix_completion_tokens + forced_think_end_tokens,
            forced_think_end_tokens=forced_think_end_tokens,
            reasoning_prefix_tokens=reasoning_prefix_tokens,
        )
        event_logger.log(
            "memory_match_topup_sample",
            method=method,
            group_size=group_size,
            prompt_index=example.prompt_index,
            sample_index=sample_index,
            correct=record.correct,
            prefix_completion_tokens=record.prefix_completion_tokens,
            completion_tokens=record.completion_tokens,
        )
        return record
    finally:
        if session_opened:
            client.close_session(session_id)


def generate_memory_match_shared_group_samples(
    *,
    args: argparse.Namespace,
    benchmark: str,
    client: SGLangRestClient,
    example: Example,
    target_dp_rank: int,
    logger: logging.Logger,
    event_logger: StructuredEventLogger,
    group_size: int,
    first_sample_index: int,
) -> List[SampleRecord]:
    reasoning_prefix_tokens = int(args.current_reasoning_prefix_tokens)
    decode_budget = continuation_token_budget(args.max_new_tokens, reasoning_prefix_tokens)
    sampling_overrides = current_sampling_overrides(args)
    parent_params = fixed_prefix_sampling_params(
        reasoning_prefix_tokens,
        sampling_overrides=sampling_overrides,
    )
    child_params = child_sampling_params(
        decode_budget,
        sampling_overrides=sampling_overrides,
    )
    method = memory_match_topup_method_name(group_size)
    group_count = int(group_size)
    group_index = int(first_sample_index) // max(group_count, 1)
    child_sample_indices = list(range(int(first_sample_index), int(first_sample_index) + group_count))
    child_seeds = [
        args.seed * 100000 + example.prompt_index * 1000 + sample_index
        for sample_index in child_sample_indices
    ]
    session_id = (
        f"memory-match-shared-g{group_size}-p{example.prompt_index}-"
        f"t{reasoning_prefix_tokens}-sample{first_sample_index}"
    )
    parent_rid = f"{session_id}-prefix"
    child_rids = [f"{session_id}-child-{sample_index}" for sample_index in child_sample_indices]
    session_opened = False
    try:
        client.open_session(args.capacity_of_str_len, session_id=session_id)
        session_opened = True
        parent_output = client.generate(
            input_ids=example.prompt_ids,
            sampling_params=parent_params,
            rid=parent_rid,
            data_parallel_rank=target_dp_rank,
            session_params={"id": session_id},
        )
        if not isinstance(parent_output, dict):
            raise RuntimeError(f"Unexpected memory-match shared parent output: {parent_output!r}")
        parent_meta = parent_output.get("meta_info", {})
        parent_completion_tokens = int(parent_meta.get("completion_tokens", 0))
        prefix_reached = reached_reasoning_prefix(parent_output, reasoning_prefix_tokens)
        event_logger.log(
            "memory_match_shared_prefix",
            method=method,
            group_size=group_size,
            prompt_index=example.prompt_index,
            first_sample_index=first_sample_index,
            prefix_reached=prefix_reached,
            completion_tokens=parent_completion_tokens,
            finish_reason=parent_meta.get("finish_reason"),
        )
        if not prefix_reached:
            logger.warning(
                "[memory-match] Prompt %s group %s shared top-up at sample %s did not reach prefix.",
                example.prompt_index,
                group_size,
                first_sample_index,
            )
            return []

        parent_text = reconstruct_assistant_text(
            example.assistant_prefill,
            parent_output.get("text", ""),
        )
        fork_info = client.fork_request(
            session_id=session_id,
            parent_rid=parent_rid,
            child_count=group_count,
            child_rids=child_rids,
            child_seeds=child_seeds,
            target_dp_rank=target_dp_rank,
            allow_non_eot_branch=True,
            force_think_end=True,
        )
        if not fork_info.get("success", False):
            event_logger.log(
                "memory_match_shared_fork_failed",
                method=method,
                group_size=group_size,
                prompt_index=example.prompt_index,
                first_sample_index=first_sample_index,
                message=fork_info.get("message", "fork_request failed"),
            )
            return []
        forced_think_end_tokens = int(fork_info.get("forced_think_end_token_count", 0))
        branch_parent_text = (
            parent_text + THINK_END_TAG
            if forced_think_end_tokens and THINK_END_TAG not in parent_text
            else parent_text
        )
        outputs = normalize_generate_outputs(
            client.generate(
                input_ids=[fork_info["branch_input_ids"]] * group_count,
                sampling_params=[dict(child_params) for _ in range(group_count)],
                rid=child_rids,
                data_parallel_rank=target_dp_rank,
            ),
            group_count,
            "memory-match shared top-up",
        )
        records: List[SampleRecord] = []
        for offset, output in enumerate(outputs):
            sample_index = child_sample_indices[offset]
            prefix_cost_for_sample = (
                parent_completion_tokens + forced_think_end_tokens
                if offset == 0
                else 0
            )
            forced_cost_for_sample = forced_think_end_tokens if offset == 0 else 0
            record = make_sample_record(
                benchmark=benchmark,
                method=method,
                example=example,
                sample_index=sample_index,
                rid=child_rids[offset],
                target_dp_rank=target_dp_rank,
                output=output,
                planned_seed=child_seeds[offset],
                max_new_tokens=decode_budget,
                accepted_attempt_index=None,
                prefix_text=branch_parent_text,
                prefix_completion_tokens=prefix_cost_for_sample,
                forced_think_end_tokens=forced_cost_for_sample,
                branch_group_index=group_index,
                reasoning_prefix_tokens=reasoning_prefix_tokens,
            )
            records.append(record)
            event_logger.log(
                "memory_match_shared_topup_sample",
                method=method,
                group_size=group_size,
                prompt_index=example.prompt_index,
                sample_index=sample_index,
                group_index=group_index,
                correct=record.correct,
                prefix_completion_tokens=record.prefix_completion_tokens,
                completion_tokens=record.completion_tokens,
            )
        return records
    finally:
        if session_opened:
            client.close_session(session_id)


def run_memory_match_prompt_task(
    *,
    args: argparse.Namespace,
    benchmark: str,
    host: str,
    port: int,
    api_key: Optional[str],
    timeout: int,
    example: Example,
    target_dp_rank: int,
    logger: logging.Logger,
    group_size: int,
    topup_group_size: int,
    target_cost_tokens: float,
    target_correct: bool,
    before_cost_tokens: float,
    before_correct: bool,
    existing_topup_rows: List[Dict[str, Any]],
) -> MemoryMatchPromptTaskResult:
    client = make_rest_client(host=host, port=port, api_key=api_key, timeout=timeout)
    event_logger = BufferedEventLogger()
    generated_records: List[SampleRecord] = []
    existing_by_index = {
        int(row.get("sample_index", 0)): row for row in existing_topup_rows
    }
    cumulative_cost = float(before_cost_tokens) + sum(
        sample_token_cost(row) for row in existing_by_index.values()
    )
    next_sample_index = max([args.max_k - 1] + list(existing_by_index.keys())) + 1
    status = "completed"
    message = ""
    try:
        while cumulative_cost < float(target_cost_tokens):
            if args.memory_match_topup_generator == "shared_group":
                records = generate_memory_match_shared_group_samples(
                    args=args,
                    benchmark=benchmark,
                    client=client,
                    example=example,
                    target_dp_rank=target_dp_rank,
                    logger=logger,
                    event_logger=event_logger,
                    group_size=topup_group_size,
                    first_sample_index=next_sample_index,
                )
                if not records:
                    status = "incomplete"
                    message = f"shared top-up group at sample {next_sample_index} failed"
                    break
                generated_records.extend(records)
                cumulative_cost += sum(sample_token_cost(asdict(record)) for record in records)
                next_sample_index += len(records)
            else:
                record = generate_memory_match_fixed_sample(
                    args=args,
                    benchmark=benchmark,
                    client=client,
                    example=example,
                    target_dp_rank=target_dp_rank,
                    logger=logger,
                    event_logger=event_logger,
                    group_size=group_size,
                    sample_index=next_sample_index,
                )
                if record is None:
                    status = "incomplete"
                    message = f"top-up sample {next_sample_index} failed"
                    break
                generated_records.append(record)
                cumulative_cost += int(record.prefix_completion_tokens) + int(record.completion_tokens)
                next_sample_index += 1
            if len(generated_records) > max(args.max_k * 8, 256):
                status = "incomplete"
                message = "safety guard hit while memory matching"
                break
        topup_rows = list(existing_by_index.values()) + [asdict(record) for record in generated_records]
        prompt_row = build_memory_match_prompt_row(
            group_size=group_size,
            topup_generator=str(args.memory_match_topup_generator),
            topup_group_size=topup_group_size,
            prompt_index=example.prompt_index,
            problem_id=example.problem_id,
            base_k=args.max_k,
            target_cost_tokens=target_cost_tokens,
            target_correct=target_correct,
            before_cost_tokens=before_cost_tokens,
            before_correct=before_correct,
            topup_records=topup_rows,
            status=status,
            message=message,
        )
        return MemoryMatchPromptTaskResult(
            group_size=group_size,
            topup_group_size=topup_group_size,
            prompt_index=example.prompt_index,
            target_dp_rank=target_dp_rank,
            prompt_row=prompt_row,
            sample_records=generated_records,
            event_records=list(event_logger.records),
        )
    finally:
        client.close()


def write_memory_match_artifacts(
    output_dir: Path,
    prompt_rows: List[Dict[str, Any]],
    run_timing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    summary_rows: List[Dict[str, Any]] = []
    group_sizes = sorted({int(row["group_size"]) for row in prompt_rows})
    for group_size in group_sizes:
        rows = [
            row
            for row in prompt_rows
            if int(row["group_size"]) == group_size and row.get("status") == "completed"
        ]
        if not rows:
            continue
        summary_rows.append(
            {
                "group_size": group_size,
                "num_prompts": len(rows),
                "before_accuracy": sum(1 for row in rows if row.get("before_correct")) / len(rows),
                "after_accuracy": sum(1 for row in rows if row.get("after_correct")) / len(rows),
                "target_cost_tokens": sum(float(row["target_cost_tokens"]) for row in rows) / len(rows),
                "before_cost_tokens": sum(float(row["before_cost_tokens"]) for row in rows) / len(rows),
                "after_cost_tokens": sum(float(row["after_cost_tokens"]) for row in rows) / len(rows),
                "effective_k": sum(float(row["effective_k"]) for row in rows) / len(rows),
                "topup_count": sum(float(row["topup_count"]) for row in rows) / len(rows),
                "reached_target_rate": sum(1 for row in rows if row.get("reached_target")) / len(rows),
            }
        )
    write_csv(output_dir / "memory_match_prompts.csv", prompt_rows)
    write_csv(output_dir / "summary_memory_match.csv", summary_rows)
    payload: Dict[str, Any] = {
        "prompt_rows": prompt_rows,
        "summary_rows": summary_rows,
    }
    if run_timing is not None:
        payload["run_timing"] = run_timing
    (output_dir / "summary_memory_match.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["# Memory-Matched Top-Up", ""]
    if run_timing and run_timing.get("wall_clock_seconds") is not None:
        lines.append(
            f"- duration: {run_timing.get('wall_clock_hms') or format_wall_clock_seconds(run_timing.get('wall_clock_seconds'))}"
        )
        lines.append("")
    lines.append("| Shared Group | Before Acc. | After Acc. | Before Memory | After Memory | Target Memory | Effective k |")
    lines.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in summary_rows:
        lines.append(
            f"| {row['group_size']} | {row['before_accuracy']:.4f} | {row['after_accuracy']:.4f} | "
            f"{row['before_cost_tokens']:.1f} | {row['after_cost_tokens']:.1f} | "
            f"{row['target_cost_tokens']:.1f} | {row['effective_k']:.2f} |"
        )
    (output_dir / "summary_memory_match.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    return payload


def memory_match_checkpoint_config(
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[Optional[str], Optional[Path], Optional[str], int]:
    family = str(getattr(args, "memory_match_checkpoint_family", "") or "")
    if not family:
        return None, None, None, 0
    root_value = str(getattr(args, "memory_match_checkpoint_root", "") or "")
    if root_value:
        root = Path(root_value).resolve()
    else:
        repeat_dir = output_dir.parent.parent if output_dir.parent.name.startswith("shard") else output_dir.parent
        root = repeat_dir.parent if repeat_dir.name.startswith("repeat_") else output_dir.parent
    campaign = str(getattr(args, "memory_match_checkpoint_campaign", "") or "")
    if not campaign:
        campaign = output_dir.parent.name if output_dir.parent.name.startswith("memory_match_topup") else output_dir.name
    step = max(1, int(getattr(args, "memory_match_checkpoint_step", 20) or 20))
    return family, root, campaign, step


def load_checkpoint_prompt_rows(
    *,
    root: Path,
    campaign: str,
    prompt_filename: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for repeat_dir in sorted(path for path in root.glob("repeat_*") if path.is_dir()):
        repeat_index = infer_repeat_index_from_path(repeat_dir)
        for prompt_path in sorted((repeat_dir / campaign).glob(f"shard*/{prompt_filename}")):
            for row in load_jsonl(prompt_path):
                if row.get("status") != "completed":
                    continue
                out = dict(row)
                out["_repeat_index"] = repeat_index
                out["_source_path"] = str(prompt_path)
                rows.append(out)
    rows.sort(
        key=lambda row: (
            value_to_float_or_none(row.get("completed_at_unix")) or 0.0,
            int(row.get("_repeat_index") or 0),
            int(row.get("prompt_index") or 0),
        )
    )
    return rows


def summarize_checkpoint_points(
    prompt_rows: List[Dict[str, Any]],
    point_specs: List[Dict[str, str]],
) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []
    for spec in point_specs:
        condition = spec["condition"]
        cost_key = spec["cost_key"]
        correct_key = spec["correct_key"]
        marker = spec.get("marker", "o")
        color = spec.get("color", "#666666")
        repeat_buckets: Dict[int, List[Dict[str, Any]]] = {}
        for row in prompt_rows:
            cost = value_to_float_or_none(row.get(cost_key))
            correct = value_to_bool_or_none(row.get(correct_key))
            repeat_index = row.get("_repeat_index")
            if cost is None or correct is None or repeat_index is None:
                continue
            repeat_buckets.setdefault(int(repeat_index), []).append(row)

        per_repeat_rows: List[Dict[str, Any]] = []
        for repeat_index, rows in sorted(repeat_buckets.items()):
            accuracy, _, _ = mean_std_optional(
                1.0 if value_to_bool_or_none(row.get(correct_key)) else 0.0
                for row in rows
            )
            cost, _, _ = mean_std_optional(value_to_float_or_none(row.get(cost_key)) for row in rows)
            per_repeat_rows.append(
                {
                    "repeat_index": repeat_index,
                    "num_rows": len(rows),
                    "accuracy": accuracy,
                    "cost_tokens": cost,
                }
            )

        accuracy_mean, accuracy_std, n_repeats = mean_std_optional(
            value_to_float_or_none(row.get("accuracy")) for row in per_repeat_rows
        )
        cost_mean, cost_std, _ = mean_std_optional(
            value_to_float_or_none(row.get("cost_tokens")) for row in per_repeat_rows
        )
        prompt_mean, _, _ = mean_std_optional(
            value_to_float_or_none(row.get("num_rows")) for row in per_repeat_rows
        )
        if accuracy_mean is None or cost_mean is None:
            continue
        summary_rows.append(
            {
                "condition": condition,
                "mean_accuracy": accuracy_mean,
                "std_accuracy": accuracy_std,
                "mean_cost_tokens": cost_mean,
                "std_cost_tokens": cost_std,
                "num_repeats": n_repeats,
                "mean_prompt_rows_per_repeat": prompt_mean,
                "total_prompt_rows": sum(int(row["num_rows"]) for row in per_repeat_rows),
                "marker": marker,
                "color": color,
            }
        )
    return summary_rows


def write_checkpoint_scatter_plot(
    out_dir: Path,
    summary_rows: List[Dict[str, Any]],
    *,
    checkpoint_count: int,
    title: str,
    filename_stem: str,
) -> None:
    if not summary_rows:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / f"{filename_stem}.csv", summary_rows)
    (out_dir / f"{filename_stem}.json").write_text(
        json.dumps(
            {
                "checkpoint_prompt_rows": checkpoint_count,
                "summary_rows": summary_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        write_checkpoint_scatter_svg_fallback(
            out_dir / f"{filename_stem}.svg",
            summary_rows,
            checkpoint_count=checkpoint_count,
            title=title,
        )
        return

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.7,
            "figure.dpi": 180,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
        }
    )
    fig, ax = plt.subplots(figsize=(7.5, 4.9))
    for row in summary_rows:
        ax.errorbar(
            [float(row["mean_cost_tokens"])],
            [float(row["mean_accuracy"])],
            xerr=[float(row.get("std_cost_tokens") or 0.0)],
            yerr=[float(row.get("std_accuracy") or 0.0)],
            marker=str(row.get("marker") or "o"),
            color=str(row.get("color") or "#666666"),
            linestyle="none",
            markersize=7,
            capsize=3,
            label=str(row["condition"]),
        )
    ax.set_xlabel("Memory Usage (tokens)")
    ax.set_ylabel("Pass@32")
    ax.set_title(f"{title} ({checkpoint_count} prompt-repeat rows)")
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(out_dir / f"{filename_stem}.png")
    fig.savefig(out_dir / f"{filename_stem}.svg")
    plt.close(fig)


def write_checkpoint_scatter_svg_fallback(
    path: Path,
    summary_rows: List[Dict[str, Any]],
    *,
    checkpoint_count: int,
    title: str,
) -> None:
    width, height = 1080, 740
    left, right, top, bottom = 120, 800, 80, 620
    xs = [
        float(row["mean_cost_tokens"])
        for row in summary_rows
        if row.get("mean_cost_tokens") is not None
    ]
    ys = [
        float(row["mean_accuracy"])
        for row in summary_rows
        if row.get("mean_accuracy") is not None
    ]
    if not xs or not ys:
        return
    xlo, xhi = min(xs), max(xs)
    ylo, yhi = min(ys), max(ys)
    xpad = (xhi - xlo) * 0.1 or max(xhi * 0.05, 1.0)
    ypad = (yhi - ylo) * 0.15 or 0.05
    xlo -= xpad
    xhi += xpad
    ylo = max(0.0, ylo - ypad)
    yhi = min(1.0, yhi + ypad)

    def sx(value: float) -> float:
        return left + (value - xlo) / max(xhi - xlo, 1e-9) * (right - left)

    def sy(value: float) -> float:
        return bottom - (value - ylo) / max(yhi - ylo, 1e-9) * (bottom - top)

    def token_label(value: float) -> str:
        return f"{value / 1000:.0f}k" if abs(value) >= 1000 else f"{value:.0f}"

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="42" text-anchor="middle" font-family="serif" font-size="24">{title} ({checkpoint_count} prompt-repeat rows)</text>',
    ]
    for idx in range(6):
        x_value = xlo + (xhi - xlo) * idx / 5
        x = sx(x_value)
        parts.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{x:.1f}" y="{bottom + 28}" text-anchor="middle" font-family="serif" font-size="15">{token_label(x_value)}</text>')
        y_value = ylo + (yhi - ylo) * idx / 5
        y = sy(y_value)
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e5e5e5"/>')
        parts.append(f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end" font-family="serif" font-size="15">{y_value:.2f}</text>')
    parts.append(f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222"/>')
    parts.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222"/>')
    parts.append(f'<text x="{(left + right) / 2:.1f}" y="{height - 38}" text-anchor="middle" font-family="serif" font-size="19">Memory Usage (tokens)</text>')
    parts.append(f'<text transform="translate(38 {(top + bottom) / 2:.1f}) rotate(-90)" text-anchor="middle" font-family="serif" font-size="19">Pass@32</text>')
    legend_x, legend_y = 830, 95
    for idx, row in enumerate(summary_rows):
        color = str(row.get("color") or "#666666")
        x = sx(float(row["mean_cost_tokens"]))
        y = sy(float(row["mean_accuracy"]))
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6" fill="{color}"/>')
        ly = legend_y + idx * 30
        parts.append(f'<circle cx="{legend_x}" cy="{ly}" r="6" fill="{color}"/>')
        parts.append(f'<text x="{legend_x + 20}" y="{ly + 5}" font-family="serif" font-size="15">{str(row["condition"])}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def maybe_write_memory_match_checkpoint_plots(
    args: argparse.Namespace,
    output_dir: Path,
    *,
    prompt_filename: str = "memory_match_prompts.jsonl",
) -> None:
    family, root, campaign, step = memory_match_checkpoint_config(args, output_dir)
    if not family or root is None or not campaign:
        return
    completed_rows = load_checkpoint_prompt_rows(
        root=root,
        campaign=campaign,
        prompt_filename=prompt_filename,
    )
    if len(completed_rows) < step:
        return
    if family in {"shared2", "shared_topup"}:
        before_label = (
            str(getattr(args, "memory_match_checkpoint_before_label", "") or "")
            or ("Shared every 2" if family == "shared2" else "Shared")
        )
        after_label = (
            str(getattr(args, "memory_match_checkpoint_after_label", "") or "")
            or f"{before_label} + top-up"
        )
        point_specs = [
            {"condition": "Fixed", "cost_key": "target_cost_tokens", "correct_key": "target_correct", "marker": "o", "color": "#4C78A8"},
            {"condition": before_label, "cost_key": "before_cost_tokens", "correct_key": "before_correct", "marker": "s", "color": "#F58518"},
            {"condition": after_label, "cost_key": "after_cost_tokens", "correct_key": "after_correct", "marker": "D", "color": "#E45756"},
        ]
        title = (
            str(getattr(args, "memory_match_checkpoint_title", "") or "")
            or ("AIME Train: Shared Every 2 Top-Up" if family == "shared2" else "AIME Train: Shared Top-Up")
        )
    elif family == "mixed_fixed_shared":
        point_specs = [
            {"condition": "Fixed", "cost_key": "target_cost_tokens", "correct_key": "target_correct", "marker": "o", "color": "#4C78A8"},
            {"condition": "Shared", "cost_key": "shared_before_cost_tokens", "correct_key": "shared_before_correct", "marker": "^", "color": "#54A24B"},
            {"condition": "Fixed + Shared", "cost_key": "mixed_before_cost_tokens", "correct_key": "mixed_before_correct", "marker": "D", "color": "#E45756"},
            {"condition": "Shared + fixed top-up", "cost_key": "shared_after_cost_tokens", "correct_key": "shared_after_correct", "marker": "v", "color": "#72B7B2"},
            {"condition": "Fixed + Shared + fixed top-up", "cost_key": "mixed_after_cost_tokens", "correct_key": "mixed_after_correct", "marker": "P", "color": "#B279A2"},
        ]
        title = "AIME Train: Fixed/Shared Top-Up"
    else:
        return

    checkpoint_dir = root / "checkpoint_plots" / campaign
    for checkpoint_count in range(step, len(completed_rows) + 1, step):
        filename_stem = f"checkpoint_{checkpoint_count:04d}_memory_vs_pass32"
        if (checkpoint_dir / f"{filename_stem}.json").exists():
            continue
        summary_rows = summarize_checkpoint_points(
            completed_rows[:checkpoint_count],
            point_specs,
        )
        write_checkpoint_scatter_plot(
            checkpoint_dir,
            summary_rows,
            checkpoint_count=checkpoint_count,
            title=title,
            filename_stem=filename_stem,
        )


def mixed_topup_method_name() -> str:
    return "memory_match_fixed_shared_fixed_topup"


def load_mixed_topup_sources(
    source_root: Path,
    *,
    repeat_index: int,
    max_k: int,
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    aggregate_dir = source_root / "aggregate"
    if not (aggregate_dir / "per_prompt_pure.csv").exists() and source_root.name.startswith("repeat_"):
        aggregate_dir = source_root.parent / "aggregate"
    pure_rows = read_csv_dict_rows(aggregate_dir / "per_prompt_pure.csv")
    mixed_rows = read_csv_dict_rows(aggregate_dir / "per_prompt_mixed.csv")
    fixed_by_prompt: Dict[int, Dict[str, Any]] = {}
    shared_by_prompt: Dict[int, Dict[str, Any]] = {}
    mixed_by_prompt: Dict[int, Dict[str, Any]] = {}
    for row in pure_rows:
        if int(float(row.get("repeat_index") or -1)) != repeat_index:
            continue
        if int(float(row.get("k") or 0)) != max_k:
            continue
        prompt_index = int(float(row["prompt_index"]))
        condition = str(row.get("condition") or "")
        if condition == "Fixed_32":
            fixed_by_prompt[prompt_index] = row
        elif condition == "Shared_32":
            shared_by_prompt[prompt_index] = row
    for row in mixed_rows:
        if int(float(row.get("repeat_index") or -1)) != repeat_index:
            continue
        if int(float(row.get("k") or 0)) != max_k:
            continue
        prompt_index = int(float(row["prompt_index"]))
        if str(row.get("condition") or "") == "Mixed Fixed+Shared":
            mixed_by_prompt[prompt_index] = row
    return fixed_by_prompt, shared_by_prompt, mixed_by_prompt


def apply_topup_records_until_target(
    *,
    before_cost_tokens: float,
    before_correct: bool,
    target_cost_tokens: float,
    topup_records: List[Dict[str, Any]],
) -> Tuple[float, bool, int, bool]:
    cumulative_cost = float(before_cost_tokens)
    cumulative_correct = bool(before_correct)
    selected_count = 0
    for record in sorted(topup_records, key=lambda row: int(row.get("sample_index", 0))):
        if cumulative_cost >= float(target_cost_tokens):
            break
        cumulative_cost += sample_token_cost(record)
        cumulative_correct = cumulative_correct or bool(record.get("correct", False))
        selected_count += 1
    return cumulative_cost, cumulative_correct, selected_count, cumulative_cost >= float(target_cost_tokens)


def build_mixed_memory_match_prompt_row(
    *,
    prompt_index: int,
    problem_id: str,
    target_cost_tokens: float,
    target_correct: bool,
    shared_before_cost_tokens: float,
    shared_before_correct: bool,
    mixed_before_cost_tokens: float,
    mixed_before_correct: bool,
    topup_records: List[Dict[str, Any]],
    base_k: int,
    status: str = "completed",
    message: str = "",
) -> Dict[str, Any]:
    shared_after_cost, shared_after_correct, shared_topup_count, shared_reached = apply_topup_records_until_target(
        before_cost_tokens=shared_before_cost_tokens,
        before_correct=shared_before_correct,
        target_cost_tokens=target_cost_tokens,
        topup_records=topup_records,
    )
    mixed_after_cost, mixed_after_correct, mixed_topup_count, mixed_reached = apply_topup_records_until_target(
        before_cost_tokens=mixed_before_cost_tokens,
        before_correct=mixed_before_correct,
        target_cost_tokens=target_cost_tokens,
        topup_records=topup_records,
    )
    return {
        "prompt_index": int(prompt_index),
        "problem_id": problem_id,
        "target_cost_tokens": float(target_cost_tokens),
        "target_correct": bool(target_correct),
        "shared_before_cost_tokens": float(shared_before_cost_tokens),
        "shared_before_correct": bool(shared_before_correct),
        "mixed_before_cost_tokens": float(mixed_before_cost_tokens),
        "mixed_before_correct": bool(mixed_before_correct),
        "shared_after_cost_tokens": float(shared_after_cost),
        "shared_after_correct": bool(shared_after_correct),
        "mixed_after_cost_tokens": float(mixed_after_cost),
        "mixed_after_correct": bool(mixed_after_correct),
        "shared_topup_count": int(shared_topup_count),
        "mixed_topup_count": int(mixed_topup_count),
        "shared_effective_k": int(base_k) + int(shared_topup_count),
        "mixed_effective_k": int(base_k) + int(mixed_topup_count),
        "generated_topup_count": len(topup_records),
        "shared_reached_target": bool(shared_reached),
        "mixed_reached_target": bool(mixed_reached),
        "status": status,
        "message": message,
        "completed_at_unix": time.time(),
    }


def run_mixed_memory_match_prompt_task(
    *,
    args: argparse.Namespace,
    benchmark: str,
    host: str,
    port: int,
    api_key: Optional[str],
    timeout: int,
    example: Example,
    target_dp_rank: int,
    logger: logging.Logger,
    target_cost_tokens: float,
    target_correct: bool,
    shared_before_cost_tokens: float,
    shared_before_correct: bool,
    mixed_before_cost_tokens: float,
    mixed_before_correct: bool,
    existing_topup_rows: List[Dict[str, Any]],
) -> MemoryMatchPromptTaskResult:
    client = make_rest_client(host=host, port=port, api_key=api_key, timeout=timeout)
    event_logger = BufferedEventLogger()
    generated_records: List[SampleRecord] = []
    existing_by_index = {
        int(row.get("sample_index", 0)): row for row in existing_topup_rows
    }
    shared_cost, _, _, _ = apply_topup_records_until_target(
        before_cost_tokens=shared_before_cost_tokens,
        before_correct=shared_before_correct,
        target_cost_tokens=target_cost_tokens,
        topup_records=list(existing_by_index.values()),
    )
    next_sample_index = max([args.max_k - 1] + list(existing_by_index.keys())) + 1
    status = "completed"
    message = ""
    try:
        while shared_cost < float(target_cost_tokens):
            record = generate_memory_match_fixed_sample(
                args=args,
                benchmark=benchmark,
                client=client,
                example=example,
                target_dp_rank=target_dp_rank,
                logger=logger,
                event_logger=event_logger,
                group_size=args.max_k,
                sample_index=next_sample_index,
            )
            if record is None:
                status = "incomplete"
                message = f"fixed top-up sample {next_sample_index} failed"
                break
            generated_records.append(record)
            shared_cost += int(record.prefix_completion_tokens) + int(record.completion_tokens)
            next_sample_index += 1
            if len(existing_by_index) + len(generated_records) > max(args.max_k * 10, 320):
                status = "incomplete"
                message = "safety guard hit while fixed/shared memory matching"
                break
        topup_rows = list(existing_by_index.values()) + [asdict(record) for record in generated_records]
        prompt_row = build_mixed_memory_match_prompt_row(
            prompt_index=example.prompt_index,
            problem_id=example.problem_id,
            base_k=args.max_k,
            target_cost_tokens=target_cost_tokens,
            target_correct=target_correct,
            shared_before_cost_tokens=shared_before_cost_tokens,
            shared_before_correct=shared_before_correct,
            mixed_before_cost_tokens=mixed_before_cost_tokens,
            mixed_before_correct=mixed_before_correct,
            topup_records=topup_rows,
            status=status,
            message=message,
        )
        return MemoryMatchPromptTaskResult(
            group_size=args.max_k,
            topup_group_size=args.max_k,
            prompt_index=example.prompt_index,
            target_dp_rank=target_dp_rank,
            prompt_row=prompt_row,
            sample_records=generated_records,
            event_records=list(event_logger.records),
        )
    finally:
        client.close()


def write_mixed_memory_match_artifacts(
    output_dir: Path,
    prompt_rows: List[Dict[str, Any]],
    run_timing: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    completed = [row for row in prompt_rows if row.get("status") == "completed"]
    specs = [
        ("Fixed", "target_cost_tokens", "target_correct", "", ""),
        ("Shared", "shared_before_cost_tokens", "shared_before_correct", "", ""),
        ("Fixed + Shared", "mixed_before_cost_tokens", "mixed_before_correct", "", ""),
        ("Shared + fixed top-up", "shared_after_cost_tokens", "shared_after_correct", "shared_topup_count", "shared_reached_target"),
        ("Fixed + Shared + fixed top-up", "mixed_after_cost_tokens", "mixed_after_correct", "mixed_topup_count", "mixed_reached_target"),
    ]
    summary_rows: List[Dict[str, Any]] = []
    for condition, cost_key, correct_key, topup_key, reached_key in specs:
        rows = [
            row
            for row in completed
            if value_to_float_or_none(row.get(cost_key)) is not None
            and value_to_bool_or_none(row.get(correct_key)) is not None
        ]
        if not rows:
            continue
        summary_rows.append(
            {
                "condition": condition,
                "num_prompts": len(rows),
                "accuracy": sum(1 for row in rows if value_to_bool_or_none(row.get(correct_key))) / len(rows),
                "cost_tokens": sum(float(row[cost_key]) for row in rows) / len(rows),
                "topup_count": (
                    sum(float(row.get(topup_key) or 0.0) for row in rows) / len(rows)
                    if topup_key
                    else 0.0
                ),
                "reached_target_rate": (
                    sum(1 for row in rows if value_to_bool_or_none(row.get(reached_key))) / len(rows)
                    if reached_key
                    else 1.0
                ),
            }
        )
    write_csv(output_dir / "mixed_memory_match_prompts.csv", prompt_rows)
    write_csv(output_dir / "summary_mixed_memory_match.csv", summary_rows)
    payload: Dict[str, Any] = {
        "prompt_rows": prompt_rows,
        "summary_rows": summary_rows,
    }
    if run_timing is not None:
        payload["run_timing"] = run_timing
    (output_dir / "summary_mixed_memory_match.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    lines = ["# Fixed/Shared Memory-Matched Top-Up", ""]
    lines.append("| Condition | Accuracy | Memory | Top-up Count | Reached Target | Prompts |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for row in summary_rows:
        lines.append(
            f"| {row['condition']} | {row['accuracy']:.4f} | {row['cost_tokens']:.1f} | "
            f"{row['topup_count']:.2f} | {row['reached_target_rate']:.4f} | {row['num_prompts']} |"
        )
    (output_dir / "summary_mixed_memory_match.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    return payload


def run_mixed_memory_match_topup_experiment(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    runtime_config: RuntimeConfig,
    scheduler_config: SchedulerConfig,
    pass_at_ks: List[int],
    examples: List[Example],
    prompt_build_info: PromptBuildInfo,
    client: SGLangRestClient,
    multiplex_self_check: Optional[MultiplexSelfCheckResult],
    run_started_at_unix: float,
) -> Dict[str, Any]:
    source_root = Path(args.memory_match_source_root).resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"memory_match_source_root does not exist: {source_root}")
    repeat_index = infer_repeat_index_from_path(output_dir)
    if repeat_index is None:
        raise RuntimeError(f"Could not infer repeat_XX from output_dir: {output_dir}")
    reasoning_prefix_tokens = int(args.branch_ablation_reasoning_prefix_tokens)
    args.current_reasoning_prefix_tokens = reasoning_prefix_tokens
    logger = configure_logging(output_dir, logger_name="compare_passk_aime.mixed_memory_match_topup")
    events_path = output_dir / "events.jsonl"
    event_logger = StructuredEventLogger(events_path)
    samples_path = output_dir / "topup_samples.jsonl"
    prompt_rows_path = output_dir / "mixed_memory_match_prompts.jsonl"
    sample_rows = load_jsonl(samples_path) if args.resume else []
    prompt_rows = load_jsonl(prompt_rows_path) if args.resume else []
    if not args.resume and (samples_path.exists() or prompt_rows_path.exists()):
        raise RuntimeError(f"{output_dir} already contains mixed memory-match outputs. Use --resume or a fresh directory.")

    fixed_rows, shared_rows, mixed_rows = load_mixed_topup_sources(
        source_root,
        repeat_index=repeat_index,
        max_k=args.max_k,
    )
    write_manifest(
        output_dir,
        args,
        runtime_config,
        scheduler_config,
        pass_at_ks,
        examples,
        [mixed_topup_method_name()],
        prompt_build_info,
        multiplex_self_check=multiplex_self_check,
        run_status="running",
        run_timing=build_run_timing(run_started_at_unix),
        reasoning_prefix_tokens=reasoning_prefix_tokens,
        reasoning_prefix_token_values=[reasoning_prefix_tokens],
    )

    completed_prompts = {
        int(row["prompt_index"])
        for row in prompt_rows
        if row.get("status") == "completed"
    }
    sample_rows_by_prompt: Dict[int, List[Dict[str, Any]]] = {}
    for row in sample_rows:
        sample_rows_by_prompt.setdefault(int(row.get("prompt_index")), []).append(row)

    task_items: List[Dict[str, Any]] = []
    for example in examples:
        if example.prompt_index in completed_prompts:
            continue
        fixed = fixed_rows.get(example.prompt_index)
        shared = shared_rows.get(example.prompt_index)
        mixed = mixed_rows.get(example.prompt_index)
        if fixed is None or shared is None or mixed is None:
            prompt_rows.append(
                {
                    "prompt_index": example.prompt_index,
                    "problem_id": example.problem_id,
                    "status": "missing_source",
                    "message": "missing fixed/shared/mixed source row",
                    "completed_at_unix": time.time(),
                }
            )
            continue
        fixed_cost = value_to_float_or_none(fixed.get("cost_tokens"))
        fixed_correct = value_to_bool_or_none(fixed.get("correct"))
        shared_cost = value_to_float_or_none(shared.get("cost_tokens"))
        shared_correct = value_to_bool_or_none(shared.get("correct"))
        mixed_cost = value_to_float_or_none(mixed.get("cost_tokens"))
        mixed_correct = value_to_bool_or_none(mixed.get("correct"))
        if None in (fixed_cost, fixed_correct, shared_cost, shared_correct, mixed_cost, mixed_correct):
            prompt_rows.append(
                {
                    "prompt_index": example.prompt_index,
                    "problem_id": example.problem_id,
                    "status": "missing_source",
                    "message": f"missing cost/pass@{args.max_k}",
                    "completed_at_unix": time.time(),
                }
            )
            continue
        task_items.append(
            {
                "example": example,
                "target_cost_tokens": float(fixed_cost),
                "target_correct": bool(fixed_correct),
                "shared_before_cost_tokens": float(shared_cost),
                "shared_before_correct": bool(shared_correct),
                "mixed_before_cost_tokens": float(mixed_cost),
                "mixed_before_correct": bool(mixed_correct),
                "existing_topup_rows": sample_rows_by_prompt.get(example.prompt_index, []),
            }
        )

    def submit_task(task_item: Dict[str, Any], target_dp_rank: int) -> MemoryMatchPromptTaskResult:
        return run_mixed_memory_match_prompt_task(
            args=args,
            benchmark=args.benchmark,
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            timeout=args.server_timeout_seconds,
            example=task_item["example"],
            target_dp_rank=target_dp_rank,
            logger=logger,
            target_cost_tokens=float(task_item["target_cost_tokens"]),
            target_correct=bool(task_item["target_correct"]),
            shared_before_cost_tokens=float(task_item["shared_before_cost_tokens"]),
            shared_before_correct=bool(task_item["shared_before_correct"]),
            mixed_before_cost_tokens=float(task_item["mixed_before_cost_tokens"]),
            mixed_before_correct=bool(task_item["mixed_before_correct"]),
            existing_topup_rows=list(task_item.get("existing_topup_rows") or []),
        )

    def handle_result(result: MemoryMatchPromptTaskResult) -> None:
        if result.event_records:
            append_records(events_path, result.event_records)
        if result.sample_records:
            serialized = []
            for record in result.sample_records:
                row = asdict(record)
                row["topup_family"] = "mixed_fixed_shared"
                serialized.append(row)
            append_records(samples_path, serialized)
            sample_rows.extend(serialized)
        append_records(prompt_rows_path, [result.prompt_row])
        prompt_rows.append(result.prompt_row)
        write_mixed_memory_match_artifacts(
            output_dir,
            prompt_rows,
            run_timing=build_run_timing(run_started_at_unix, time.time()),
        )
        maybe_write_memory_match_checkpoint_plots(
            args,
            output_dir,
            prompt_filename="mixed_memory_match_prompts.jsonl",
        )

    scheduler_metadata = run_prompt_tasks_concurrently(
        task_items=task_items,
        scheduler_config=scheduler_config,
        effective_dp_size=runtime_config.effective_dp_size,
        submit_task=submit_task,
        handle_result=handle_result,
        logger=logger,
        event_logger=event_logger,
    )
    run_timing = build_run_timing(run_started_at_unix, time.time())
    summary_json = write_mixed_memory_match_artifacts(output_dir, prompt_rows, run_timing=run_timing)
    write_manifest(
        output_dir,
        args,
        runtime_config,
        scheduler_config,
        pass_at_ks,
        examples,
        [mixed_topup_method_name()],
        prompt_build_info,
        multiplex_self_check=multiplex_self_check,
        summary_json=summary_json,
        run_status="completed",
        run_timing=run_timing,
        reasoning_prefix_tokens=reasoning_prefix_tokens,
        reasoning_prefix_token_values=[reasoning_prefix_tokens],
        scheduler_metadata=scheduler_metadata,
    )
    return summary_json


def run_memory_match_topup_experiment(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    runtime_config: RuntimeConfig,
    scheduler_config: SchedulerConfig,
    pass_at_ks: List[int],
    examples: List[Example],
    prompt_build_info: PromptBuildInfo,
    client: SGLangRestClient,
    multiplex_self_check: Optional[MultiplexSelfCheckResult],
    run_started_at_unix: float,
) -> Dict[str, Any]:
    source_root = Path(args.memory_match_source_root).resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"memory_match_source_root does not exist: {source_root}")
    reasoning_prefix_tokens = int(args.branch_ablation_reasoning_prefix_tokens)
    args.current_reasoning_prefix_tokens = reasoning_prefix_tokens
    logger = configure_logging(output_dir, logger_name="compare_passk_aime.memory_match_topup")
    events_path = output_dir / "events.jsonl"
    event_logger = StructuredEventLogger(events_path)
    samples_path = output_dir / "topup_samples.jsonl"
    prompt_rows_path = output_dir / "memory_match_prompts.jsonl"
    sample_rows = load_jsonl(samples_path) if args.resume else []
    prompt_rows = load_jsonl(prompt_rows_path) if args.resume else []
    if not args.resume and (samples_path.exists() or prompt_rows_path.exists()):
        raise RuntimeError(f"{output_dir} already contains memory-match outputs. Use --resume or a fresh directory.")

    fixed_dirs = find_condition_dirs(source_root, "fixed_trace_independent")
    fixed_summary = load_condition_summaries_for_memory_match(
        fixed_dirs,
        pass_at_ks=pass_at_ks,
        max_k=args.max_k,
    )
    fixed_rows = per_prompt_row_lookup(fixed_summary, "baseline_independent")
    group_sizes = parse_positive_int_values(
        args.memory_match_shared_groups,
        label="memory-match shared group",
    )
    configured_topup_group_size = int(getattr(args, "memory_match_topup_shared_group_size", 0) or 0)
    if args.memory_match_topup_generator == "shared_group" and configured_topup_group_size < 0:
        raise ValueError("--memory-match-topup-shared-group-size must be non-negative")
    shared_rows_by_group: Dict[int, Dict[int, Dict[str, Any]]] = {}
    for group_size in group_sizes:
        group_dirs = find_condition_dirs(source_root, f"shared_every_{group_size:02d}")
        group_summary = load_condition_summaries_for_memory_match(
            group_dirs,
            pass_at_ks=pass_at_ks,
            max_k=args.max_k,
        )
        shared_rows_by_group[group_size] = per_prompt_row_lookup(
            group_summary,
            shared_group_method_name(group_size),
        )

    write_manifest(
        output_dir,
        args,
        runtime_config,
        scheduler_config,
        pass_at_ks,
        examples,
        [memory_match_topup_method_name(group_size) for group_size in group_sizes],
        prompt_build_info,
        multiplex_self_check=multiplex_self_check,
        run_status="running",
        run_timing=build_run_timing(run_started_at_unix),
        reasoning_prefix_tokens=reasoning_prefix_tokens,
        reasoning_prefix_token_values=[reasoning_prefix_tokens],
    )

    completed_pairs = {
        (int(row["group_size"]), int(row["prompt_index"]))
        for row in prompt_rows
        if row.get("status") == "completed"
    }
    sample_rows_by_key: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for row in sample_rows:
        group_size = int(row.get("shared_group_size") or row.get("branch_group_index") or 0)
        prompt_index = int(row.get("prompt_index"))
        sample_rows_by_key.setdefault((group_size, prompt_index), []).append(row)

    examples_by_prompt = {example.prompt_index: example for example in examples}
    task_items: List[Dict[str, Any]] = []
    for group_size in group_sizes:
        topup_group_size = (
            configured_topup_group_size
            if args.memory_match_topup_generator == "shared_group" and configured_topup_group_size > 0
            else group_size
        )
        shared_lookup = shared_rows_by_group[group_size]
        for example in examples:
            if (group_size, example.prompt_index) in completed_pairs:
                continue
            fixed_row = fixed_rows.get(example.prompt_index)
            shared_row = shared_lookup.get(example.prompt_index)
            if fixed_row is None or shared_row is None:
                prompt_rows.append(
                    build_memory_match_prompt_row(
                        group_size=group_size,
                        prompt_index=example.prompt_index,
                        problem_id=example.problem_id,
                        base_k=args.max_k,
                        target_cost_tokens=None,
                        target_correct=None,
                        before_cost_tokens=None,
                        before_correct=None,
                        topup_records=[],
                        status="missing_source",
                        message="missing fixed or shared summary row",
                    )
                )
                continue
            target_cost = fixed_row.get(f"cost_at_{args.max_k}")
            target_correct = fixed_row.get(f"pass_at_{args.max_k}")
            before_cost = shared_row.get(f"cost_at_{args.max_k}")
            before_correct = shared_row.get(f"pass_at_{args.max_k}")
            if target_cost is None or target_correct is None or before_cost is None or before_correct is None:
                prompt_rows.append(
                    build_memory_match_prompt_row(
                        group_size=group_size,
                        prompt_index=example.prompt_index,
                        problem_id=example.problem_id,
                        base_k=args.max_k,
                        target_cost_tokens=None,
                        target_correct=None,
                        before_cost_tokens=None,
                        before_correct=None,
                        topup_records=[],
                        status="missing_source",
                        message=f"missing cost/pass@{args.max_k}",
                    )
                )
                continue
            task_items.append(
                {
                    "example": example,
                    "group_size": group_size,
                    "topup_group_size": topup_group_size,
                    "target_cost_tokens": float(target_cost),
                    "target_correct": bool(target_correct),
                    "before_cost_tokens": float(before_cost),
                    "before_correct": bool(before_correct),
                    "existing_topup_rows": sample_rows_by_key.get((group_size, example.prompt_index), []),
                }
            )

    def submit_task(task_item: Dict[str, Any], target_dp_rank: int) -> MemoryMatchPromptTaskResult:
        return run_memory_match_prompt_task(
            args=args,
            benchmark=args.benchmark,
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            timeout=args.server_timeout_seconds,
            example=task_item["example"],
            target_dp_rank=target_dp_rank,
            logger=logger,
            group_size=int(task_item["group_size"]),
            topup_group_size=int(task_item["topup_group_size"]),
            target_cost_tokens=float(task_item["target_cost_tokens"]),
            target_correct=bool(task_item["target_correct"]),
            before_cost_tokens=float(task_item["before_cost_tokens"]),
            before_correct=bool(task_item["before_correct"]),
            existing_topup_rows=list(task_item.get("existing_topup_rows") or []),
        )

    def handle_result(result: MemoryMatchPromptTaskResult) -> None:
        if result.event_records:
            append_records(events_path, result.event_records)
        if result.sample_records:
            serialized = []
            for record in result.sample_records:
                row = asdict(record)
                row["shared_group_size"] = int(result.group_size)
                row["topup_shared_group_size"] = int(result.topup_group_size)
                serialized.append(row)
            append_records(samples_path, serialized)
            sample_rows.extend(serialized)
        append_records(prompt_rows_path, [result.prompt_row])
        prompt_rows.append(result.prompt_row)
        write_memory_match_artifacts(
            output_dir,
            prompt_rows,
            run_timing=build_run_timing(run_started_at_unix, time.time()),
        )
        maybe_write_memory_match_checkpoint_plots(args, output_dir)

    scheduler_metadata = run_prompt_tasks_concurrently(
        task_items=task_items,
        scheduler_config=scheduler_config,
        effective_dp_size=runtime_config.effective_dp_size,
        submit_task=submit_task,
        handle_result=handle_result,
        logger=logger,
        event_logger=event_logger,
    )
    run_timing = build_run_timing(run_started_at_unix, time.time())
    summary_json = write_memory_match_artifacts(output_dir, prompt_rows, run_timing=run_timing)
    write_manifest(
        output_dir,
        args,
        runtime_config,
        scheduler_config,
        pass_at_ks,
        examples,
        [memory_match_topup_method_name(group_size) for group_size in group_sizes],
        prompt_build_info,
        multiplex_self_check=multiplex_self_check,
        summary_json=summary_json,
        run_status="completed",
        run_timing=run_timing,
        reasoning_prefix_tokens=reasoning_prefix_tokens,
        reasoning_prefix_token_values=[reasoning_prefix_tokens],
        scheduler_metadata=scheduler_metadata,
    )
    return summary_json


def adaptive_method_name(shared_count: int) -> str:
    return f"adaptive_shared_{int(shared_count)}_fixed_topup"


def adaptive_condition_dir(root_output_dir: Path, shared_count: int) -> Path:
    return root_output_dir / f"adaptive_shared_{int(shared_count):02d}"


def answer_agreement_confidence(records: List[SampleRecord]) -> Tuple[float, Optional[str], Dict[str, int]]:
    answers: List[str] = []
    for record in records:
        answer = normalize_candidate_text(record.extracted_answer or "")
        if answer:
            answers.append(answer)
    if not records or not answers:
        return 0.0, None, {}
    counts = Counter(answers)
    majority_answer, majority_count = counts.most_common(1)[0]
    return majority_count / max(len(records), 1), majority_answer, dict(counts)


def make_zero_cost_padding_sample(
    *,
    benchmark: str,
    method: str,
    example: Example,
    sample_index: int,
    target_dp_rank: int,
    reasoning_prefix_tokens: int,
) -> SampleRecord:
    return SampleRecord(
        benchmark=benchmark,
        method=method,
        prompt_index=example.prompt_index,
        problem_id=example.problem_id,
        sample_index=sample_index,
        rid=f"{method}-p{example.prompt_index}-padding-{sample_index}",
        target_dp_rank=target_dp_rank,
        correct=False,
        score=0.0,
        text="",
        finish_reason={"type": "adaptive_stop", "matched": "confidence_threshold"},
        prompt_tokens=0,
        completion_tokens=0,
        cached_tokens=0,
        latency_seconds=0.0,
        prefix_completion_tokens=0,
        usable_for_eval=True,
        excluded_reason=None,
        score_reason="adaptive_no_fixed_topup",
        extracted_answer=None,
        score_debug={"adaptive_padding": True},
        generated_suffix="",
        reasoning_prefix_tokens=reasoning_prefix_tokens,
    )


def run_adaptive_shared_prefix_for_prompt(
    *,
    args: argparse.Namespace,
    benchmark: str,
    client: SGLangRestClient,
    example: Example,
    target_dp_rank: int,
    logger: logging.Logger,
    event_logger: StructuredEventLogger,
    shared_count: int,
    method: str,
) -> Tuple[List[SampleRecord], List[Dict[str, Any]], Dict[str, Any]]:
    reasoning_prefix_tokens = int(args.current_reasoning_prefix_tokens)
    decode_budget = continuation_token_budget(args.max_new_tokens, reasoning_prefix_tokens)
    sampling_overrides = current_sampling_overrides(args)
    parent_params = fixed_prefix_sampling_params(
        reasoning_prefix_tokens,
        sampling_overrides=sampling_overrides,
    )
    child_params = child_sampling_params(
        decode_budget,
        sampling_overrides=sampling_overrides,
    )
    child_seeds = planned_child_seeds(args.seed, example.prompt_index, args.max_k)
    session_id = (
        f"adaptive-shared{shared_count}-p{example.prompt_index}-"
        f"t{reasoning_prefix_tokens}"
    )
    parent_rid = f"{session_id}-prefix"
    child_rids = [f"{session_id}-child-{index}" for index in range(shared_count)]
    slot_statuses: List[Dict[str, Any]] = [
        {
            "slot_index": slot_index,
            "source": "shared",
            "prefix_completion_tokens": 0,
            "prefix_latency_seconds": 0.0,
            "prefix_finish_reason": None,
            "prefix_reached": False,
            "forced_think_end_tokens": 0,
            "continuation_completion_tokens": 0,
            "continuation_latency_seconds": 0.0,
            "accepted_for_eval": False,
            "failure_reason": None,
        }
        for slot_index in range(shared_count)
    ]
    records: List[SampleRecord] = []
    metadata: Dict[str, Any] = {
        "shared_prefix_completion_tokens": 0,
        "shared_forced_think_end_tokens": 0,
        "shared_prefix_reached": False,
        "shared_attempted": True,
        "shared_success": False,
        "shared_failure_reason": None,
    }
    logger.info(
        "[adaptive-ablation] Prompt %s: generating Shared@%s with prefix=%s.",
        example.prompt_index,
        shared_count,
        reasoning_prefix_tokens,
    )
    event_logger.log(
        "adaptive_shared_prompt_start",
        method=method,
        prompt_index=example.prompt_index,
        shared_count=shared_count,
        target_dp_rank=target_dp_rank,
        reasoning_prefix_tokens=reasoning_prefix_tokens,
    )
    session_opened = False
    try:
        client.open_session(args.capacity_of_str_len, session_id=session_id)
        session_opened = True
        parent_output = client.generate(
            input_ids=example.prompt_ids,
            sampling_params=parent_params,
            rid=parent_rid,
            data_parallel_rank=target_dp_rank,
            session_params={"id": session_id},
        )
        if not isinstance(parent_output, dict):
            raise RuntimeError(f"Unexpected adaptive parent output: {parent_output!r}")
        parent_meta = parent_output.get("meta_info", {})
        parent_completion_tokens = int(parent_meta.get("completion_tokens", 0))
        parent_latency_seconds = float(parent_meta.get("e2e_latency", 0.0))
        parent_finish_reason = parent_meta.get("finish_reason")
        parent_text = reconstruct_assistant_text(
            example.assistant_prefill,
            parent_output.get("text", ""),
        )
        prefix_reached = reached_reasoning_prefix(parent_output, reasoning_prefix_tokens)
        metadata.update(
            {
                "shared_prefix_completion_tokens": parent_completion_tokens,
                "shared_prefix_reached": prefix_reached,
                "shared_parent_finish_reason": parent_finish_reason,
            }
        )
        for status in slot_statuses:
            status["prefix_completion_tokens"] = parent_completion_tokens
            status["prefix_latency_seconds"] = parent_latency_seconds
            status["prefix_finish_reason"] = parent_finish_reason
            status["prefix_reached"] = prefix_reached
        if not prefix_reached:
            for status in slot_statuses:
                status["failure_reason"] = "prefix_not_reached"
            metadata["shared_failure_reason"] = "prefix_not_reached"
            event_logger.log(
                "adaptive_shared_prefix_not_reached",
                method=method,
                prompt_index=example.prompt_index,
                shared_count=shared_count,
                completion_tokens=parent_completion_tokens,
                finish_reason=parent_finish_reason,
            )
            return records, slot_statuses, metadata

        fork_info = client.fork_request(
            session_id=session_id,
            parent_rid=parent_rid,
            child_count=shared_count,
            child_rids=child_rids,
            child_seeds=child_seeds[:shared_count],
            target_dp_rank=target_dp_rank,
            allow_non_eot_branch=True,
            force_think_end=True,
        )
        if not fork_info.get("success", False):
            for status in slot_statuses:
                status["failure_reason"] = "fork_failed"
            metadata["shared_failure_reason"] = "fork_failed"
            event_logger.log(
                "adaptive_shared_fork_failed",
                method=method,
                prompt_index=example.prompt_index,
                shared_count=shared_count,
                message=fork_info.get("message", "fork_request failed"),
            )
            return records, slot_statuses, metadata
        forced_think_end_tokens = int(fork_info.get("forced_think_end_token_count", 0))
        metadata["shared_forced_think_end_tokens"] = forced_think_end_tokens
        branch_parent_text = (
            parent_text + THINK_END_TAG
            if forced_think_end_tokens and THINK_END_TAG not in parent_text
            else parent_text
        )
        outputs = normalize_generate_outputs(
            client.generate(
                input_ids=[fork_info["branch_input_ids"]] * shared_count,
                sampling_params=[dict(child_params) for _ in range(shared_count)],
                rid=child_rids,
                data_parallel_rank=target_dp_rank,
            ),
            shared_count,
            "adaptive shared children batch",
        )
        for sample_index, output in enumerate(outputs):
            continuation_meta = output.get("meta_info", {})
            slot_status = slot_statuses[sample_index]
            slot_status["forced_think_end_tokens"] = forced_think_end_tokens
            slot_status["continuation_completion_tokens"] = int(
                continuation_meta.get("completion_tokens", 0)
            )
            slot_status["continuation_latency_seconds"] = float(
                continuation_meta.get("e2e_latency", 0.0)
            )
            slot_status["accepted_for_eval"] = True
            record = make_sample_record(
                benchmark=benchmark,
                method=method,
                example=example,
                sample_index=sample_index,
                rid=child_rids[sample_index],
                target_dp_rank=target_dp_rank,
                output=output,
                planned_seed=child_seeds[sample_index],
                max_new_tokens=decode_budget,
                accepted_attempt_index=None,
                prefix_text=branch_parent_text,
                prefix_completion_tokens=(
                    parent_completion_tokens + forced_think_end_tokens
                    if sample_index == 0
                    else 0
                ),
                forced_think_end_tokens=forced_think_end_tokens if sample_index == 0 else 0,
                reasoning_prefix_tokens=reasoning_prefix_tokens,
            )
            records.append(record)
            event_logger.log(
                "adaptive_shared_child_sample",
                method=method,
                prompt_index=example.prompt_index,
                shared_count=shared_count,
                sample_index=sample_index,
                correct=record.correct,
                extracted_answer=record.extracted_answer,
                prefix_completion_tokens=record.prefix_completion_tokens,
                completion_tokens=record.completion_tokens,
            )
        metadata["shared_success"] = len(records) == shared_count
        return records, slot_statuses, metadata
    finally:
        if session_opened:
            client.close_session(session_id)


def run_adaptive_ablation_for_prompt(
    *,
    args: argparse.Namespace,
    benchmark: str,
    client: SGLangRestClient,
    example: Example,
    target_dp_rank: int,
    logger: logging.Logger,
    event_logger: StructuredEventLogger,
    shared_count: int,
    confidence_threshold: float,
) -> Tuple[BaselinePromptResult, List[SampleRecord]]:
    reasoning_prefix_tokens = int(args.current_reasoning_prefix_tokens)
    method = adaptive_method_name(shared_count)
    shared_records, shared_slot_statuses, shared_metadata = run_adaptive_shared_prefix_for_prompt(
        args=args,
        benchmark=benchmark,
        client=client,
        example=example,
        target_dp_rank=target_dp_rank,
        logger=logger,
        event_logger=event_logger,
        shared_count=shared_count,
        method=method,
    )
    confidence, majority_answer, answer_counts = answer_agreement_confidence(shared_records)
    trigger_fixed = confidence < confidence_threshold and len(shared_records) < args.max_k
    existing_indices = {int(record.sample_index) for record in shared_records}
    fixed_records: List[SampleRecord] = []
    fixed_slot_statuses: List[Dict[str, Any]] = []
    if trigger_fixed:
        logger.info(
            "[adaptive-ablation] Prompt %s Shared@%s confidence %.3f < %.3f; adding %s fixed samples.",
            example.prompt_index,
            shared_count,
            confidence,
            confidence_threshold,
            args.max_k - len(existing_indices),
        )
        _, baseline_records, _ = run_baseline_for_prompt(
            args=args,
            benchmark=benchmark,
            client=client,
            example=example,
            target_dp_rank=target_dp_rank,
            logger=logger,
            event_logger=event_logger,
            existing_sample_indices=existing_indices,
        )
        fixed_records = [replace(record, method=method) for record in baseline_records]
    else:
        logger.info(
            "[adaptive-ablation] Prompt %s Shared@%s confidence %.3f >= %.3f; skipping fixed top-up.",
            example.prompt_index,
            shared_count,
            confidence,
            confidence_threshold,
        )

    records = sorted(shared_records + fixed_records, key=lambda record: int(record.sample_index))
    current_indices = {int(record.sample_index) for record in records}
    padding_records: List[SampleRecord] = []
    if not trigger_fixed:
        for sample_index in range(args.max_k):
            if sample_index not in current_indices:
                padding_records.append(
                    make_zero_cost_padding_sample(
                        benchmark=benchmark,
                        method=method,
                        example=example,
                        sample_index=sample_index,
                        target_dp_rank=target_dp_rank,
                        reasoning_prefix_tokens=reasoning_prefix_tokens,
                    )
                )
    records = sorted(records + padding_records, key=lambda record: int(record.sample_index))
    fixed_indices = sorted(
        int(record.sample_index)
        for record in fixed_records
    )
    slot_statuses: List[Dict[str, Any]] = []
    shared_status_by_slot = {int(status["slot_index"]): status for status in shared_slot_statuses}
    for sample_index in range(args.max_k):
        if sample_index in shared_status_by_slot:
            status = dict(shared_status_by_slot[sample_index])
            status["source"] = "shared"
        elif sample_index in fixed_indices:
            record = next(record for record in fixed_records if int(record.sample_index) == sample_index)
            status = {
                "slot_index": sample_index,
                "source": "fixed_topup",
                "prefix_completion_tokens": int(record.prefix_completion_tokens)
                - int(record.forced_think_end_tokens),
                "prefix_latency_seconds": 0.0,
                "prefix_finish_reason": None,
                "prefix_reached": True,
                "forced_think_end_tokens": int(record.forced_think_end_tokens),
                "continuation_completion_tokens": int(record.completion_tokens),
                "continuation_latency_seconds": float(record.latency_seconds),
                "accepted_for_eval": True,
                "failure_reason": None,
            }
        else:
            status = {
                "slot_index": sample_index,
                "source": "adaptive_skipped_fixed",
                "prefix_completion_tokens": 0,
                "prefix_latency_seconds": 0.0,
                "prefix_finish_reason": None,
                "prefix_reached": True,
                "forced_think_end_tokens": 0,
                "continuation_completion_tokens": 0,
                "continuation_latency_seconds": 0.0,
                "accepted_for_eval": True,
                "failure_reason": None,
            }
        slot_statuses.append(status)

    success = len(records) >= args.max_k
    reject_reason_counts_payload = dict(
        Counter(
            str(status["failure_reason"])
            for status in slot_statuses
            if status.get("failure_reason")
        )
    )
    prompt_result = BaselinePromptResult(
        benchmark=benchmark,
        prompt_index=example.prompt_index,
        problem_id=example.problem_id,
        target_dp_rank=target_dp_rank,
        success=success,
        message=(
            f"Adaptive Shared@{shared_count}; confidence={confidence:.3f}; "
            f"fixed_topup={'yes' if trigger_fixed else 'no'}."
        ),
        required_sample_count=args.max_k,
        usable_sample_count=len(records),
        attempts_used=1 + len(fixed_records),
        total_completion_tokens_spent=sum(
            int(record.prefix_completion_tokens) + int(record.completion_tokens)
            for record in records
        ),
        total_latency_seconds_spent=sum(
            float(status["prefix_latency_seconds"]) + float(status["continuation_latency_seconds"])
            for status in slot_statuses
        ),
        reject_reason_counts=reject_reason_counts_payload,
        slot_statuses=slot_statuses,
        reasoning_prefix_tokens=reasoning_prefix_tokens,
        method=method,
    )
    event_logger.log(
        "adaptive_prompt_done",
        method=method,
        prompt_index=example.prompt_index,
        shared_count=shared_count,
        confidence=confidence,
        confidence_threshold=confidence_threshold,
        majority_answer=majority_answer,
        answer_counts=answer_counts,
        trigger_fixed=trigger_fixed,
        shared_samples=len(shared_records),
        fixed_samples=len(fixed_records),
        padding_samples=len(padding_records),
        total_completion_tokens_spent=prompt_result.total_completion_tokens_spent,
        shared_metadata=shared_metadata,
    )
    return prompt_result, records


def run_adaptive_ablation_prompt_task(
    *,
    args: argparse.Namespace,
    benchmark: str,
    host: str,
    port: int,
    api_key: Optional[str],
    timeout: int,
    example: Example,
    target_dp_rank: int,
    logger: logging.Logger,
    shared_count: int,
    confidence_threshold: float,
) -> StandardPromptTaskResult:
    client = make_rest_client(host=host, port=port, api_key=api_key, timeout=timeout)
    event_logger = BufferedEventLogger()
    try:
        prompt_result, sample_records = run_adaptive_ablation_for_prompt(
            args=args,
            benchmark=benchmark,
            client=client,
            example=example,
            target_dp_rank=target_dp_rank,
            logger=logger,
            event_logger=event_logger,
            shared_count=shared_count,
            confidence_threshold=confidence_threshold,
        )
        return StandardPromptTaskResult(
            prompt_index=example.prompt_index,
            target_dp_rank=target_dp_rank,
            prompt_result=prompt_result,
            sample_records=sample_records,
            attempt_records=[],
            event_records=list(event_logger.records),
        )
    finally:
        client.close()


def run_adaptive_ablation_count_experiment(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    runtime_config: RuntimeConfig,
    scheduler_config: SchedulerConfig,
    pass_at_ks: List[int],
    examples: List[Example],
    prompt_build_info: PromptBuildInfo,
    multiplex_self_check: Optional[MultiplexSelfCheckResult],
    reasoning_prefix_tokens: int,
    shared_count: int,
    confidence_threshold: float,
) -> Dict[str, Any]:
    run_started_at_unix = time.time()
    logger = configure_logging(
        output_dir,
        logger_name=f"compare_passk_aime.adaptive_{shared_count}",
    )
    events_path = output_dir / "events.jsonl"
    event_logger = StructuredEventLogger(events_path)
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

    args.current_reasoning_prefix_tokens = reasoning_prefix_tokens
    method = adaptive_method_name(shared_count)
    write_manifest(
        output_dir,
        args,
        runtime_config,
        scheduler_config,
        pass_at_ks,
        examples,
        [method],
        prompt_build_info,
        multiplex_self_check=multiplex_self_check,
        run_status="running",
        run_timing=build_run_timing(run_started_at_unix),
        reasoning_prefix_tokens=reasoning_prefix_tokens,
        reasoning_prefix_token_values=[reasoning_prefix_tokens],
    )
    completed_pairs = determine_completed_prompts(prompt_rows, [])
    task_items = [
        {"example": example}
        for example in examples
        if not completed_pairs.get((method, example.prompt_index), False)
    ]

    def submit_task(task_item: Dict[str, Any], target_dp_rank: int) -> StandardPromptTaskResult:
        return run_adaptive_ablation_prompt_task(
            args=args,
            benchmark=args.benchmark,
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            timeout=args.server_timeout_seconds,
            example=task_item["example"],
            target_dp_rank=target_dp_rank,
            logger=logger,
            shared_count=shared_count,
            confidence_threshold=confidence_threshold,
        )

    def handle_result(result: StandardPromptTaskResult) -> None:
        if result.event_records:
            append_records(events_path, result.event_records)
        if result.prompt_result is not None:
            serialized_result = asdict(result.prompt_result)
            append_records(prompt_rows_path, [serialized_result])
            prompt_rows.append(serialized_result)
            completed_pairs[(method, result.prompt_index)] = True
        if result.sample_records:
            serialized_records = [asdict(record) for record in result.sample_records]
            append_records(samples_path, serialized_records)
            sample_rows.extend(serialized_records)
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

    scheduler_metadata = run_prompt_tasks_concurrently(
        task_items=task_items,
        scheduler_config=scheduler_config,
        effective_dp_size=runtime_config.effective_dp_size,
        submit_task=submit_task,
        handle_result=handle_result,
        logger=logger,
        event_logger=event_logger,
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
        scheduler_config,
        pass_at_ks,
        examples,
        [method],
        prompt_build_info,
        multiplex_self_check=multiplex_self_check,
        summary_json=summary_json,
        run_status="completed",
        run_timing=run_timing,
        reasoning_prefix_tokens=reasoning_prefix_tokens,
        reasoning_prefix_token_values=[reasoning_prefix_tokens],
        scheduler_metadata=scheduler_metadata,
    )
    return summary_json


def technique_color(method: str) -> str:
    if method == "baseline_independent":
        return "#1f77b4"
    if method.startswith("shared_trace"):
        return "#ff7f0e"
    if method == "standard_generation_independent":
        return "#2ca02c"
    return "#7f7f7f"


def collect_clean_rows_with_condition(
    summary: Optional[Dict[str, Any]],
    **condition: Any,
) -> List[Dict[str, Any]]:
    if summary is None:
        return []
    rows = []
    for row in summary.get("clean_summary_rows", []):
        row_copy = dict(row)
        row_copy.update(condition)
        rows.append(row_copy)
    return rows


def ablation_condition_label(row: Dict[str, Any]) -> str:
    if row.get("policy") == "adaptive_fixed_topup":
        shared_count = int(row.get("adaptive_shared_count", row.get("branch_group_size", 0)) or 0)
        max_k_value = int(row.get("adaptive_max_k", 16) or 16)
        fixed_count = max(max_k_value - shared_count, 0)
        threshold = float(row.get("confidence_threshold", DEFAULT_ADAPTIVE_CONFIDENCE_THRESHOLD))
        if fixed_count <= 0:
            return f"Shared@{shared_count}"
        return f"Shared@{shared_count} + Fixed@{fixed_count} if conf<{threshold:g}"
    group_size = int(row.get("branch_group_size", 0) or 0)
    if group_size == 1:
        return "Shared every 1 / fixed trace"
    if group_size > 1:
        return f"Shared every {group_size}"
    return str(row.get("condition") or method_display_name(str(row.get("method", ""))))


def ablation_condition_sort_key(row: Dict[str, Any]) -> Tuple[int, int]:
    return (int(row.get("branch_group_size", 0) or 0), int(row.get("k", 0) or 0))


def ablation_condition_style(group_size: int) -> Dict[str, Any]:
    palette = {
        1: "#4C78A8",
        2: "#F58518",
        6: "#ECA82C",
        4: "#54A24B",
        8: "#B279A2",
        10: "#9D755D",
        12: "#BAB0AC",
        14: "#A0CBE8",
        16: "#E45756",
        32: "#72B7B2",
        64: "#EECA3B",
    }
    markers = {
        1: "o",
        2: "s",
        4: "^",
        6: "v",
        8: "D",
        10: "<",
        12: ">",
        14: "*",
        16: "P",
        32: "X",
        64: "v",
    }
    return {
        "color": palette.get(group_size, "#7f7f7f"),
        "marker": markers.get(group_size, "o"),
        "linewidth": 2.25,
        "markersize": 6.5,
    }


def write_branch_ablation_plots(
    root_output_dir: Path,
    rows: List[Dict[str, Any]],
    *,
    max_k: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    plot_rows = [
        row
        for row in rows
        if row.get("pass_at_k") is not None
        and row.get("avg_cost_tokens") is not None
        and int(row.get("k", 0) or 0) > 0
    ]
    if not plot_rows:
        return

    groups = sorted({int(row.get("branch_group_size", 0) or 0) for row in plot_rows})
    max_k_rows = [row for row in plot_rows if int(row.get("k", 0) or 0) == max_k]

    def rows_for_group(group_size: int) -> List[Dict[str, Any]]:
        return sorted(
            [row for row in plot_rows if int(row.get("branch_group_size", 0) or 0) == group_size],
            key=lambda row: int(row["k"]),
        )

    def save_k_vs_pass(path: Path, *, ylim: Optional[Tuple[float, float]], title: str) -> None:
        plt.figure(figsize=(9.5, 6.0))
        ax = plt.gca()
        for group_size in groups:
            group_rows = rows_for_group(group_size)
            if not group_rows:
                continue
            style = ablation_condition_style(group_size)
            ax.plot(
                [int(row["k"]) for row in group_rows],
                [float(row["pass_at_k"]) for row in group_rows],
                label=ablation_condition_label(group_rows[0]),
                **style,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(sorted({int(row["k"]) for row in plot_rows}))
        ax.set_xticklabels([str(value) for value in sorted({int(row["k"]) for row in plot_rows})])
        ax.set_xlabel("k")
        ax.set_ylabel("Pass@k Performance")
        ax.set_ylim(*(ylim or (0.0, 1.0)))
        ax.set_title(title)
        ax.grid(True, which="major", linestyle="--", alpha=0.32)
        ax.legend(title="Technique", frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=240)
        plt.close()

    observed_y = [float(row["pass_at_k"]) for row in plot_rows]
    y_min = max(0.0, min(observed_y) - 0.04)
    y_max = min(1.0, max(observed_y) + 0.04)
    save_k_vs_pass(
        root_output_dir / "ablation_passk_scaling_curves.png",
        ylim=(0.0, 1.0),
        title=f"Branch Ablation Pass@k Scaling (Pass@{max_k} run)",
    )
    save_k_vs_pass(
        root_output_dir / "ablation_passk_scaling_curves_zoomed.png",
        ylim=(y_min, y_max),
        title=f"Branch Ablation Pass@k Scaling (Zoomed, Pass@{max_k} run)",
    )

    plt.figure(figsize=(9.5, 6.0))
    ax = plt.gca()
    for group_size in groups:
        group_rows = rows_for_group(group_size)
        if not group_rows:
            continue
        style = ablation_condition_style(group_size)
        ax.plot(
            [float(row["avg_cost_tokens"]) for row in group_rows],
            [float(row["pass_at_k"]) for row in group_rows],
            label=ablation_condition_label(group_rows[0]),
            **style,
        )
    ax.set_xlabel("Memory Usage (tokens)")
    ax.set_ylabel("Pass@k Performance")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Branch Ablation: Memory Usage vs Pass@k")
    ax.grid(True, linestyle="--", alpha=0.32)
    ax.legend(title="Technique", frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(root_output_dir / "ablation_memory_vs_passk.png", dpi=240)
    plt.close()

    plt.figure(figsize=(9.5, 6.0))
    ax = plt.gca()
    for group_size in groups:
        group_rows = rows_for_group(group_size)
        if not group_rows:
            continue
        style = ablation_condition_style(group_size)
        ax.plot(
            [int(row["k"]) for row in group_rows],
            [float(row["avg_cost_tokens"]) for row in group_rows],
            label=ablation_condition_label(group_rows[0]),
            **style,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({int(row["k"]) for row in plot_rows}))
    ax.set_xticklabels([str(value) for value in sorted({int(row["k"]) for row in plot_rows})])
    ax.set_xlabel("k")
    ax.set_ylabel("Generated Tokens")
    ax.set_title("Branch Ablation: k vs Generated Tokens")
    ax.grid(True, which="major", linestyle="--", alpha=0.32)
    ax.legend(title="Technique", frameon=False, fontsize=8)
    plt.tight_layout()
    plt.savefig(root_output_dir / "ablation_k_vs_generated_tokens.png", dpi=240)
    plt.close()

    if max_k_rows:
        plt.figure(figsize=(9.5, 6.0))
        ax = plt.gca()
        for row in sorted(max_k_rows, key=ablation_condition_sort_key):
            group_size = int(row.get("branch_group_size", 0) or 0)
            style = ablation_condition_style(group_size)
            ax.scatter(
                [float(row["avg_cost_tokens"])],
                [float(row["pass_at_k"])],
                s=95,
                color=style["color"],
                marker=style["marker"],
                label=ablation_condition_label(row),
            )
            ax.annotate(
                ablation_condition_label(row),
                (float(row["avg_cost_tokens"]), float(row["pass_at_k"])),
                textcoords="offset points",
                xytext=(6, 6),
                fontsize=8,
            )
        ax.set_xlabel("Memory Usage (tokens)")
        ax.set_ylabel(f"Pass@{max_k} Performance")
        ax.set_ylim(0.0, 1.0)
        ax.set_title(f"Branch Ablation: Memory Usage vs Pass@{max_k}")
        ax.grid(True, linestyle="--", alpha=0.32)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            dedup = dict(zip(labels, handles))
            ax.legend(dedup.values(), dedup.keys(), frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(root_output_dir / f"ablation_memory_vs_pass{max_k}.png", dpi=240)
        plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for group_size in groups:
        group_rows = rows_for_group(group_size)
        if not group_rows:
            continue
        style = ablation_condition_style(group_size)
        label = ablation_condition_label(group_rows[0])
        axes[0].plot(
            [int(row["k"]) for row in group_rows],
            [float(row["pass_at_k"]) for row in group_rows],
            label=label,
            **style,
        )
        axes[1].plot(
            [float(row["avg_cost_tokens"]) for row in group_rows],
            [float(row["pass_at_k"]) for row in group_rows],
            label=label,
            **style,
        )
        axes[2].plot(
            [int(row["k"]) for row in group_rows],
            [float(row["avg_cost_tokens"]) for row in group_rows],
            label=label,
            **style,
        )
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(sorted({int(row["k"]) for row in plot_rows}))
    axes[0].set_xticklabels([str(value) for value in sorted({int(row["k"]) for row in plot_rows})])
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Pass@k Performance")
    axes[0].set_ylim(0.0, 1.0)
    axes[0].set_title("k vs Pass@k")
    axes[1].set_xlabel("Memory Usage (tokens)")
    axes[1].set_ylabel("Pass@k Performance")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_title("Memory Usage vs Pass@k")
    axes[2].set_xscale("log", base=2)
    axes[2].set_xticks(sorted({int(row["k"]) for row in plot_rows}))
    axes[2].set_xticklabels([str(value) for value in sorted({int(row["k"]) for row in plot_rows})])
    axes[2].set_xlabel("k")
    axes[2].set_ylabel("Generated Tokens")
    axes[2].set_title("k vs Generated Tokens")
    for ax in axes:
        ax.grid(True, which="major", linestyle="--", alpha=0.32)
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 4), frameon=False)
    fig.suptitle(f"Branch Ablation Summary (Pass@{max_k} run)", y=1.03)
    plt.tight_layout()
    plt.savefig(root_output_dir / "ablation_compact_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)

    technique_dir = root_output_dir / "technique_plots"
    technique_dir.mkdir(parents=True, exist_ok=True)
    for group_size in groups:
        group_rows = rows_for_group(group_size)
        if not group_rows:
            continue
        style = ablation_condition_style(group_size)
        label = ablation_condition_label(group_rows[0])
        safe_label = "shared_every_01" if group_size == 1 else f"shared_every_{group_size:02d}"
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.0))
        axes[0].plot(
            [int(row["k"]) for row in group_rows],
            [float(row["pass_at_k"]) for row in group_rows],
            label=label,
            **style,
        )
        axes[1].plot(
            [float(row["avg_cost_tokens"]) for row in group_rows],
            [float(row["pass_at_k"]) for row in group_rows],
            label=label,
            **style,
        )
        axes[2].plot(
            [int(row["k"]) for row in group_rows],
            [float(row["avg_cost_tokens"]) for row in group_rows],
            label=label,
            **style,
        )
        axes[0].set_xscale("log", base=2)
        axes[0].set_xticks([int(row["k"]) for row in group_rows])
        axes[0].set_xticklabels([str(int(row["k"])) for row in group_rows])
        axes[0].set_xlabel("k")
        axes[0].set_ylabel("Pass@k Performance")
        axes[0].set_ylim(0.0, 1.0)
        axes[0].set_title("k vs Pass@k")
        axes[1].set_xlabel("Memory Usage (tokens)")
        axes[1].set_ylabel("Pass@k Performance")
        axes[1].set_ylim(0.0, 1.0)
        axes[1].set_title("Memory Usage vs Pass@k")
        axes[2].set_xscale("log", base=2)
        axes[2].set_xticks([int(row["k"]) for row in group_rows])
        axes[2].set_xticklabels([str(int(row["k"])) for row in group_rows])
        axes[2].set_xlabel("k")
        axes[2].set_ylabel("Generated Tokens")
        axes[2].set_title("k vs Generated Tokens")
        for ax in axes:
            ax.grid(True, which="major", linestyle="--", alpha=0.32)
        fig.suptitle(label, y=1.03)
        plt.tight_layout()
        plt.savefig(technique_dir / f"{safe_label}_summary.png", dpi=240, bbox_inches="tight")
        plt.close(fig)


def write_branch_ablation_summary(
    root_output_dir: Path,
    *,
    baseline_summary: Optional[Dict[str, Any]],
    group_summaries: Dict[int, Dict[str, Any]],
    pass_at_ks: List[int],
    max_k: int,
    run_timing: Optional[Dict[str, Any]] = None,
) -> None:
    rows: List[Dict[str, Any]] = []
    rows.extend(
        collect_clean_rows_with_condition(
            baseline_summary,
            condition="fixed_trace_independent",
            branch_group_size=1,
        )
    )
    for group_size, summary in sorted(group_summaries.items()):
        rows.extend(
            collect_clean_rows_with_condition(
                summary,
                condition=f"shared_every_{group_size}",
                branch_group_size=group_size,
            )
        )
    write_csv(root_output_dir / "summary_ablation.csv", rows)
    payload = {
        "baseline": baseline_summary,
        "groups": {str(group_size): summary for group_size, summary in sorted(group_summaries.items())},
    }
    if run_timing is not None:
        payload["run_timing"] = run_timing
    (root_output_dir / "summary_ablation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = ["# Branch Ablation", ""]
    if run_timing and run_timing.get("wall_clock_seconds") is not None:
        lines.append(
            f"- duration: {run_timing.get('wall_clock_hms') or format_wall_clock_seconds(run_timing.get('wall_clock_seconds'))}"
        )
        lines.append("")
    lines.append(f"| Condition | Method | pass@{max_k} | Memory Usage (tokens) | Prompts |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in rows:
        if int(row.get("k", 0)) != max_k:
            continue
        pass_value = "" if row.get("pass_at_k") is None else f"{float(row['pass_at_k']):.4f}"
        cost_value = "" if row.get("avg_cost_tokens") is None else f"{float(row['avg_cost_tokens']):.1f}"
        lines.append(
            f"| {row.get('condition', '')} | {method_display_name(str(row['method']))} | "
            f"{pass_value} | {cost_value} | {row.get('num_prompts', 0)} |"
        )
    (root_output_dir / "summary_ablation.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )

    write_branch_ablation_plots(root_output_dir, rows, max_k=max_k)


def load_existing_adaptive_ablation_summaries(
    root_output_dir: Path,
    *,
    pass_at_ks: List[int],
    max_k: int,
) -> Dict[int, Dict[str, Any]]:
    summaries: Dict[int, Dict[str, Any]] = {}
    for condition_dir in sorted(root_output_dir.glob("adaptive_shared_*")):
        if not condition_dir.is_dir():
            continue
        try:
            shared_count = int(condition_dir.name.rsplit("_", 1)[1])
        except ValueError:
            continue
        summary = summarize_existing_output_dir(
            condition_dir,
            pass_at_ks=pass_at_ks,
            max_k=max_k,
            includes_shared_trace=False,
        )
        if summary is not None:
            summaries[shared_count] = summary
    return summaries


def write_adaptive_ablation_summary(
    root_output_dir: Path,
    *,
    adaptive_summaries: Dict[int, Dict[str, Any]],
    pass_at_ks: List[int],
    max_k: int,
    confidence_threshold: float,
    run_timing: Optional[Dict[str, Any]] = None,
) -> None:
    rows: List[Dict[str, Any]] = []
    for shared_count, summary in sorted(adaptive_summaries.items()):
        rows.extend(
            collect_clean_rows_with_condition(
                summary,
                condition=f"adaptive_shared_{shared_count}",
                branch_group_size=shared_count,
                adaptive_shared_count=shared_count,
                adaptive_fixed_count=max(max_k - shared_count, 0),
                adaptive_max_k=max_k,
                confidence_threshold=confidence_threshold,
                policy="adaptive_fixed_topup",
            )
        )
    write_csv(root_output_dir / "summary_adaptive_ablation.csv", rows)
    # Keep the generic ablation name too so plotting/reporting tools can find it.
    write_csv(root_output_dir / "summary_ablation.csv", rows)
    payload: Dict[str, Any] = {
        "adaptive_conditions": {
            str(shared_count): summary
            for shared_count, summary in sorted(adaptive_summaries.items())
        },
        "confidence_threshold": confidence_threshold,
    }
    if run_timing is not None:
        payload["run_timing"] = run_timing
    (root_output_dir / "summary_adaptive_ablation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = ["# Adaptive Ablation", ""]
    if run_timing and run_timing.get("wall_clock_seconds") is not None:
        lines.append(
            f"- duration: {run_timing.get('wall_clock_hms') or format_wall_clock_seconds(run_timing.get('wall_clock_seconds'))}"
        )
    lines.append(f"- confidence trigger: fixed top-up when agreement confidence < {confidence_threshold:g}")
    lines.append("")
    lines.append(f"| Condition | Method | pass@{max_k} | Memory Usage (tokens) | Prompts |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in rows:
        if int(row.get("k", 0)) != max_k:
            continue
        pass_value = "" if row.get("pass_at_k") is None else f"{float(row['pass_at_k']):.4f}"
        cost_value = "" if row.get("avg_cost_tokens") is None else f"{float(row['avg_cost_tokens']):.1f}"
        lines.append(
            f"| {ablation_condition_label(row)} | {method_display_name(str(row['method']))} | "
            f"{pass_value} | {cost_value} | {row.get('num_prompts', 0)} |"
        )
    (root_output_dir / "summary_adaptive_ablation.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    (root_output_dir / "summary_ablation.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    write_branch_ablation_plots(root_output_dir, rows, max_k=max_k)


def hyperparam_condition_dir(
    root_output_dir: Path,
    *,
    parameter_name: str,
    phase: str,
    value: float,
) -> Path:
    return root_output_dir / f"{parameter_name}_{phase}_{safe_float_label(value)}"


def write_hyperparam_summary(
    root_output_dir: Path,
    *,
    condition_summaries: Dict[Tuple[str, str, float], Dict[str, Any]],
    pass_at_ks: List[int],
    max_k: int,
    run_timing: Optional[Dict[str, Any]] = None,
) -> None:
    rows: List[Dict[str, Any]] = []
    for (parameter_name, phase, value), summary in sorted(condition_summaries.items()):
        rows.extend(
            collect_clean_rows_with_condition(
                summary,
                parameter_name=parameter_name,
                phase=phase,
                parameter_value=value,
                condition=f"{parameter_name}_{phase}_{value:g}",
            )
        )
    write_csv(root_output_dir / "summary_hyperparam.csv", rows)
    payload = {
        "conditions": {
            f"{parameter_name}/{phase}/{value:g}": summary
            for (parameter_name, phase, value), summary in sorted(condition_summaries.items())
        }
    }
    if run_timing is not None:
        payload["run_timing"] = run_timing
    (root_output_dir / "summary_hyperparam.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = ["# Hyperparameter Sweep", ""]
    lines.append("| Parameter | Phase | Value | Method | pass@8 | Memory Usage (tokens) | Prompts |")
    lines.append("| --- | --- | ---: | --- | ---: | ---: | ---: |")
    for row in rows:
        if int(row.get("k", 0)) != max_k:
            continue
        pass_value = "" if row.get("pass_at_k") is None else f"{float(row['pass_at_k']):.4f}"
        cost_value = "" if row.get("avg_cost_tokens") is None else f"{float(row['avg_cost_tokens']):.1f}"
        lines.append(
            f"| {row.get('parameter_name', '')} | {row.get('phase', '')} | "
            f"{float(row.get('parameter_value', 0.0)):g} | {method_display_name(str(row['method']))} | "
            f"{pass_value} | {cost_value} | {row.get('num_prompts', 0)} |"
        )
    (root_output_dir / "summary_hyperparam.md").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    write_hyperparam_plots(root_output_dir, rows, max_k=max_k)


def write_hyperparam_plots(
    root_output_dir: Path,
    rows: List[Dict[str, Any]],
    *,
    max_k: int,
) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FuncFormatter
    except ModuleNotFoundError:
        return
    plot_rows = [
        row
        for row in rows
        if int(row.get("k", 0)) == max_k
        and row.get("pass_at_k") is not None
        and row.get("avg_cost_tokens") is not None
    ]
    if not plot_rows:
        return

    method_styles = {
        "baseline_independent": {
            "color": "#2B5C8A",
            "marker": "o",
            "linestyle": "-",
            "label": "Fixed-Prefix Independent",
        },
        "shared_trace_branch_after_prefix": {
            "color": "#B86E2B",
            "marker": "s",
            "linestyle": "--",
            "label": "Shared Trace After Prefix",
        },
    }
    annotation_box = {
        "boxstyle": "round,pad=0.12",
        "facecolor": "white",
        "edgecolor": "none",
        "alpha": 0.9,
    }

    def token_formatter(value: float, _pos: int) -> str:
        if abs(value) >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        if abs(value) >= 1_000:
            return f"{value / 1_000:.0f}k"
        return f"{value:.0f}"

    def parameter_label(parameter_name: str) -> str:
        if parameter_name == "top_p":
            return r"top-$p$"
        return "temperature"

    def phase_label(phase: str) -> str:
        return "thinking" if phase == HYPERPARAM_PHASE_THINKING else "discrete"

    def point_value_label(row: Dict[str, Any]) -> str:
        value = float(row["parameter_value"])
        if row.get("parameter_name") == "top_p":
            return f"{value:.2f}"
        return f"{value:.1f}"

    def method_sort_key(method: str) -> int:
        if method == "baseline_independent":
            return 0
        if method == "shared_trace_branch_after_prefix":
            return 1
        return 2

    def add_method_series(
        ax: Any,
        method_rows: List[Dict[str, Any]],
        *,
        x_key: str,
        annotate_values: bool,
    ) -> None:
        method = str(method_rows[0]["method"])
        style = method_styles.get(
            method,
            {
                "color": technique_color(method),
                "marker": "o",
                "linestyle": "-",
                "label": method_display_name(method),
            },
        )
        ax.plot(
            [float(row[x_key]) for row in method_rows],
            [float(row["pass_at_k"]) for row in method_rows],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2.05,
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=0.65,
            label=style["label"],
        )
        if not annotate_values:
            return
        data_points = [
            (float(row[x_key]), float(row["pass_at_k"]))
            for row in method_rows
        ]
        display_points = ax.transData.transform(data_points)
        dpi_scale = ax.figure.dpi / 72.0
        placed_label_boxes: List[Tuple[float, float, float, float]] = list(
            getattr(ax, "_hyperparam_label_boxes", [])
        )

        def _unit(dx: float, dy: float) -> Tuple[float, float]:
            norm = (dx * dx + dy * dy) ** 0.5
            if norm <= 1e-9:
                return (1.0, 0.0)
            return (dx / norm, dy / norm)

        def _overlap_area(
            a: Tuple[float, float, float, float],
            b: Tuple[float, float, float, float],
        ) -> float:
            x_overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            y_overlap = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
            return x_overlap * y_overlap

        def _candidate_offsets(index: int) -> List[Tuple[float, float]]:
            point_count = len(display_points)
            if point_count <= 1:
                tangent_x, tangent_y = (1.0, 0.0)
            elif index == 0:
                tangent_x = float(display_points[1][0] - display_points[0][0])
                tangent_y = float(display_points[1][1] - display_points[0][1])
            elif index == point_count - 1:
                tangent_x = float(display_points[-1][0] - display_points[-2][0])
                tangent_y = float(display_points[-1][1] - display_points[-2][1])
            else:
                tangent_x = float(display_points[index + 1][0] - display_points[index - 1][0])
                tangent_y = float(display_points[index + 1][1] - display_points[index - 1][1])
            tangent = _unit(tangent_x, tangent_y)
            normal = _unit(-tangent[1], tangent[0])
            prefer_above = method == "baseline_independent"
            if prefer_above and normal[1] < 0:
                normal = (-normal[0], -normal[1])
            elif not prefer_above and normal[1] > 0:
                normal = (-normal[0], -normal[1])

            base = 11.5
            tangent_nudge = 7.5
            endpoint_nudge = 5.5 if index == 0 else (-5.5 if index == point_count - 1 else 0.0)
            endpoint_vector = (endpoint_nudge, 0.0)
            candidates = [
                (normal[0] * base + endpoint_vector[0], normal[1] * base),
                (
                    normal[0] * base + tangent[0] * tangent_nudge + endpoint_vector[0],
                    normal[1] * base + tangent[1] * tangent_nudge,
                ),
                (
                    normal[0] * base - tangent[0] * tangent_nudge + endpoint_vector[0],
                    normal[1] * base - tangent[1] * tangent_nudge,
                ),
                (normal[0] * 15.5 + endpoint_vector[0], normal[1] * 15.5),
                (
                    normal[0] * 15.5 + tangent[0] * 10.5 + endpoint_vector[0],
                    normal[1] * 15.5 + tangent[1] * 10.5,
                ),
                (
                    normal[0] * 15.5 - tangent[0] * 10.5 + endpoint_vector[0],
                    normal[1] * 15.5 - tangent[1] * 10.5,
                ),
            ]
            if prefer_above:
                candidates.extend(
                    [
                        (0.0, 14.0),
                        (10.0, 10.0),
                        (-10.0, 10.0),
                        (0.0, 23.0),
                        (14.0, 20.0),
                        (-14.0, 20.0),
                        (0.0, 32.0),
                        (22.0, 20.0),
                        (-22.0, 20.0),
                        (30.0, 8.0),
                        (-30.0, 8.0),
                    ]
                )
            else:
                candidates.extend(
                    [
                        (0.0, -14.0),
                        (10.0, -10.0),
                        (-10.0, -10.0),
                        (0.0, -23.0),
                        (14.0, -20.0),
                        (-14.0, -20.0),
                        (0.0, -32.0),
                        (22.0, -20.0),
                        (-22.0, -20.0),
                        (30.0, -8.0),
                        (-30.0, -8.0),
                    ]
                )
            return candidates

        for index, row in enumerate(method_rows):
            label = point_value_label(row)
            label_width_px = (len(label) * 9.0 + 16.0) * dpi_scale
            label_height_px = 20.0 * dpi_scale
            point_x = float(display_points[index][0])
            point_y = float(display_points[index][1])
            data_x = float(data_points[index][0])
            data_y = float(data_points[index][1])
            data_x_values = [float(point[0]) for point in data_points]
            data_x_span = max(data_x_values) - min(data_x_values) if data_x_values else 0.0
            data_close_neighbor = any(
                other_index != index
                and abs(float(data_points[other_index][0]) - data_x)
                <= max(250.0, data_x_span * 0.04)
                and abs(float(data_points[other_index][1]) - data_y) <= 0.04
                for other_index in range(len(data_points))
            )
            prefer_above = method == "baseline_independent"
            close_neighbor = any(
                other_index != index
                and abs(float(display_points[other_index][0]) - point_x) <= label_width_px * 1.55
                and abs(float(display_points[other_index][1]) - point_y) <= label_height_px * 2.2
                for other_index in range(len(display_points))
            ) or data_close_neighbor
            nearby_labels = 0
            for placed_box in placed_label_boxes:
                placed_center_x = (placed_box[0] + placed_box[2]) / 2.0
                placed_center_y = (placed_box[1] + placed_box[3]) / 2.0
                if (
                    abs(placed_center_x - point_x) <= label_width_px * 1.35
                    and abs(placed_center_y - point_y) <= label_height_px * 2.2
                ):
                    nearby_labels += 1
            stacked_direction = 1.0 if prefer_above else -1.0
            dense_offsets: List[Tuple[float, float]] = []
            if close_neighbor:
                dense_levels = (16.0, 34.0, 52.0, 34.0, 16.0)
                dense_x_offsets = (-7.0, -3.0, 0.0, 3.0, 7.0)
                dense_level_index = index % len(dense_levels)
                stacked_y = stacked_direction * dense_levels[dense_level_index]
                dense_offsets.append((dense_x_offsets[dense_level_index], stacked_y))
            elif nearby_labels:
                stacked_y = stacked_direction * (16.0 + nearby_labels * 18.0)
                dense_offsets.extend(
                    [
                        (0.0, stacked_y),
                        (16.0, stacked_y),
                        (-16.0, stacked_y),
                        (28.0, stacked_y * 0.85),
                        (-28.0, stacked_y * 0.85),
                    ]
                )
            candidates = dense_offsets if dense_offsets else _candidate_offsets(index)
            best_offset = candidates[0]
            best_score = float("inf")
            for offset in candidates:
                label_x = point_x + offset[0] * dpi_scale
                label_y = point_y + offset[1] * dpi_scale
                label_box = (
                    label_x - label_width_px / 2.0,
                    label_y - label_height_px / 2.0,
                    label_x + label_width_px / 2.0,
                    label_y + label_height_px / 2.0,
                )
                overlap = sum(_overlap_area(label_box, placed) for placed in placed_label_boxes)
                distance_penalty = 0.02 * ((offset[0] * offset[0] + offset[1] * offset[1]) ** 0.5)
                score = overlap + distance_penalty
                if score < best_score:
                    best_offset = offset
                    best_score = score
            final_x = point_x + best_offset[0] * dpi_scale
            final_y = point_y + best_offset[1] * dpi_scale
            placed_label_boxes.append(
                (
                    final_x - label_width_px / 2.0,
                    final_y - label_height_px / 2.0,
                    final_x + label_width_px / 2.0,
                    final_y + label_height_px / 2.0,
                )
            )
            setattr(ax, "_hyperparam_label_boxes", placed_label_boxes)
            ax.annotate(
                label,
                (float(row[x_key]), float(row["pass_at_k"])),
                textcoords="offset points",
                xytext=best_offset,
                fontsize=7.5,
                color=style["color"],
                bbox=annotation_box,
                ha="center",
                va="center",
                zorder=6,
                clip_on=False,
            )

    style_context = {
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "STIXGeneral", "Times New Roman"],
        "mathtext.fontset": "stix",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#222222",
        "axes.linewidth": 0.9,
        "axes.labelsize": 11,
        "axes.titlesize": 12.3,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }

    phases = [HYPERPARAM_PHASE_THINKING, HYPERPARAM_PHASE_DISCRETE]
    parameters = ["top_p", "temperature"]
    with plt.rc_context(style_context):
        for parameter_name in parameters:
            for phase in phases:
                subset = [
                    row
                    for row in plot_rows
                    if row.get("parameter_name") == parameter_name and row.get("phase") == phase
                ]
                if not subset:
                    continue
                methods = sorted({str(row["method"]) for row in subset}, key=method_sort_key)

                fig, ax = plt.subplots(figsize=(7.6, 4.8))
                for method in methods:
                    method_rows = [row for row in subset if str(row["method"]) == method]
                    method_rows.sort(key=lambda row: float(row["parameter_value"]))
                    if method_rows:
                        add_method_series(
                            ax,
                            method_rows,
                            x_key="parameter_value",
                            annotate_values=False,
                        )
                ax.set_xlabel(parameter_label(parameter_name))
                ax.set_ylabel(rf"Pass@{max_k} Performance")
                ax.set_ylim(0.0, 1.0)
                ax.set_title(
                    rf"{parameter_label(parameter_name)} {phase_label(phase)} sweep"
                )
                ax.grid(True, color="#D8D8D8", linestyle="--", linewidth=0.65, alpha=0.55)
                ax.legend(frameon=False, loc="best")
                fig.tight_layout()
                fig.savefig(
                    root_output_dir / f"hyperparam_{parameter_name}_{phase}_passk.png",
                    dpi=300,
                )
                plt.close(fig)

                fig, ax = plt.subplots(figsize=(7.8, 4.9))
                for method in methods:
                    method_rows = [row for row in subset if str(row["method"]) == method]
                    method_rows.sort(key=lambda row: float(row["avg_cost_tokens"]))
                    if method_rows:
                        add_method_series(
                            ax,
                            method_rows,
                            x_key="avg_cost_tokens",
                            annotate_values=True,
                        )
                ax.set_xlabel("Memory Usage (tokens)")
                ax.set_ylabel(rf"Pass@{max_k} Performance")
                ax.set_ylim(0.0, 1.0)
                ax.xaxis.set_major_formatter(FuncFormatter(token_formatter))
                ax.set_title(
                    rf"{parameter_label(parameter_name)} {phase_label(phase)}: "
                    rf"Memory Usage vs Pass@{max_k}"
                )
                ax.grid(True, color="#D8D8D8", linestyle="--", linewidth=0.65, alpha=0.55)
                ax.legend(frameon=False, loc="best")
                fig.tight_layout()
                fig.savefig(
                    root_output_dir / f"hyperparam_{parameter_name}_{phase}_memory.png",
                    dpi=300,
                )
                plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(13.7, 9.2), sharey=True)
        legend_handles: List[Any] = []
        legend_labels: List[str] = []
        for row_index, parameter_name in enumerate(parameters):
            for col_index, phase in enumerate(phases):
                ax = axes[row_index][col_index]
                subset = [
                    row
                    for row in plot_rows
                    if row.get("parameter_name") == parameter_name and row.get("phase") == phase
                ]
                methods = sorted({str(row["method"]) for row in subset}, key=method_sort_key)
                for method in methods:
                    method_rows = [row for row in subset if str(row["method"]) == method]
                    method_rows.sort(key=lambda row: float(row["avg_cost_tokens"]))
                    if not method_rows:
                        continue
                    add_method_series(
                        ax,
                        method_rows,
                        x_key="avg_cost_tokens",
                        annotate_values=True,
                    )
                ax.set_title(rf"{parameter_label(parameter_name)} / {phase_label(phase)}")
                ax.set_xlabel("Memory Usage (tokens)")
                if col_index == 0:
                    ax.set_ylabel(rf"Pass@{max_k} Performance")
                ax.set_ylim(0.0, 1.0)
                ax.xaxis.set_major_formatter(FuncFormatter(token_formatter))
                ax.grid(True, color="#D8D8D8", linestyle="--", linewidth=0.65, alpha=0.55)
                handles, labels = ax.get_legend_handles_labels()
                for handle, label in zip(handles, labels):
                    if label not in legend_labels:
                        legend_handles.append(handle)
                        legend_labels.append(label)
        fig.suptitle(
            rf"Overall Hyperparameter Sweep: Memory Usage vs Pass@{max_k}",
            y=0.992,
            fontsize=16.5,
        )
        if legend_handles:
            fig.legend(
                legend_handles,
                legend_labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 0.948),
                ncol=2,
                frameon=False,
            )
        fig.tight_layout(rect=(0, 0, 1, 0.885))
        fig.savefig(
            root_output_dir / "hyperparam_overall_memory_vs_passk.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close(fig)


def run_branch_ablation_experiment(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    runtime_config: RuntimeConfig,
    scheduler_config: SchedulerConfig,
    pass_at_ks: List[int],
    examples: List[Example],
    prompt_build_info: PromptBuildInfo,
    tokenizer: Any,
    client: SGLangRestClient,
    multiplex_self_check: Optional[MultiplexSelfCheckResult],
    run_started_at_unix: float,
) -> Dict[str, Any]:
    reasoning_prefix_tokens = int(args.branch_ablation_reasoning_prefix_tokens)
    group_sizes = (
        []
        if not str(args.branch_ablation_group_sizes).strip()
        else parse_positive_int_values(
            args.branch_ablation_group_sizes,
            label="branch ablation group size",
        )
    )
    if not group_sizes and bool(getattr(args, "branch_ablation_no_baseline", False)):
        raise ValueError(
            "--branch-ablation-group-sizes cannot be empty when --branch-ablation-no-baseline is set."
        )
    args.current_reasoning_prefix_tokens = reasoning_prefix_tokens
    baseline_dir = output_dir / "fixed_trace_independent"
    baseline_summary: Optional[Dict[str, Any]] = None
    if not bool(getattr(args, "branch_ablation_no_baseline", False)):
        baseline_summary = run_fixed_prefix_experiment(
            args=args,
            output_dir=baseline_dir,
            runtime_config=runtime_config,
            scheduler_config=scheduler_config,
            pass_at_ks=pass_at_ks,
            examples=examples,
            methods=["baseline_independent"],
            prompt_build_info=prompt_build_info,
            tokenizer=tokenizer,
            client=client,
            multiplex_self_check=multiplex_self_check,
            reasoning_prefix_tokens=reasoning_prefix_tokens,
        )
    else:
        baseline_summary = summarize_existing_output_dir(
            baseline_dir,
            pass_at_ks=pass_at_ks,
            max_k=args.max_k,
            includes_shared_trace=False,
        )
    group_summaries = load_existing_branch_ablation_group_summaries(
        output_dir,
        pass_at_ks=pass_at_ks,
        max_k=args.max_k,
    )
    write_branch_ablation_summary(
        output_dir,
        baseline_summary=baseline_summary,
        group_summaries=group_summaries,
        pass_at_ks=pass_at_ks,
        max_k=args.max_k,
        run_timing=build_run_timing(run_started_at_unix, time.time()),
    )
    for group_size in group_sizes:
        group_dir = output_dir / f"shared_every_{group_size:02d}"
        group_summaries[group_size] = run_branch_ablation_group_experiment(
            args=args,
            output_dir=group_dir,
            runtime_config=runtime_config,
            scheduler_config=scheduler_config,
            pass_at_ks=pass_at_ks,
            examples=examples,
            prompt_build_info=prompt_build_info,
            multiplex_self_check=multiplex_self_check,
            reasoning_prefix_tokens=reasoning_prefix_tokens,
            group_size=group_size,
        )
        write_branch_ablation_summary(
            output_dir,
            baseline_summary=baseline_summary,
            group_summaries=group_summaries,
            pass_at_ks=pass_at_ks,
            max_k=args.max_k,
            run_timing=build_run_timing(run_started_at_unix, time.time()),
        )
    write_branch_ablation_summary(
        output_dir,
        baseline_summary=baseline_summary,
        group_summaries=group_summaries,
        pass_at_ks=pass_at_ks,
        max_k=args.max_k,
        run_timing=build_run_timing(run_started_at_unix, time.time()),
    )
    return {
        "baseline": baseline_summary,
        "groups": group_summaries,
    }


def run_adaptive_ablation_experiment(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    runtime_config: RuntimeConfig,
    scheduler_config: SchedulerConfig,
    pass_at_ks: List[int],
    examples: List[Example],
    prompt_build_info: PromptBuildInfo,
    tokenizer: Any,
    client: SGLangRestClient,
    multiplex_self_check: Optional[MultiplexSelfCheckResult],
    run_started_at_unix: float,
) -> Dict[str, Any]:
    del tokenizer
    reasoning_prefix_tokens = int(args.branch_ablation_reasoning_prefix_tokens)
    shared_counts = parse_positive_int_values(
        args.adaptive_ablation_shared_counts,
        label="adaptive ablation shared count",
    )
    if any(shared_count > int(args.max_k) for shared_count in shared_counts):
        raise ValueError("--adaptive-ablation-shared-counts cannot exceed --max-k.")
    confidence_threshold = float(args.adaptive_confidence_threshold)
    args.current_reasoning_prefix_tokens = reasoning_prefix_tokens
    adaptive_summaries = load_existing_adaptive_ablation_summaries(
        output_dir,
        pass_at_ks=pass_at_ks,
        max_k=args.max_k,
    )
    write_adaptive_ablation_summary(
        output_dir,
        adaptive_summaries=adaptive_summaries,
        pass_at_ks=pass_at_ks,
        max_k=args.max_k,
        confidence_threshold=confidence_threshold,
        run_timing=build_run_timing(run_started_at_unix, time.time()),
    )
    for shared_count in shared_counts:
        condition_dir = adaptive_condition_dir(output_dir, shared_count)
        adaptive_summaries[shared_count] = run_adaptive_ablation_count_experiment(
            args=args,
            output_dir=condition_dir,
            runtime_config=runtime_config,
            scheduler_config=scheduler_config,
            pass_at_ks=pass_at_ks,
            examples=examples,
            prompt_build_info=prompt_build_info,
            multiplex_self_check=multiplex_self_check,
            reasoning_prefix_tokens=reasoning_prefix_tokens,
            shared_count=shared_count,
            confidence_threshold=confidence_threshold,
        )
        write_adaptive_ablation_summary(
            output_dir,
            adaptive_summaries=adaptive_summaries,
            pass_at_ks=pass_at_ks,
            max_k=args.max_k,
            confidence_threshold=confidence_threshold,
            run_timing=build_run_timing(run_started_at_unix, time.time()),
        )
    write_adaptive_ablation_summary(
        output_dir,
        adaptive_summaries=adaptive_summaries,
        pass_at_ks=pass_at_ks,
        max_k=args.max_k,
        confidence_threshold=confidence_threshold,
        run_timing=build_run_timing(run_started_at_unix, time.time()),
    )
    return {
        "adaptive_conditions": adaptive_summaries,
        "confidence_threshold": confidence_threshold,
    }


def run_hyperparam_experiment(
    *,
    args: argparse.Namespace,
    output_dir: Path,
    runtime_config: RuntimeConfig,
    scheduler_config: SchedulerConfig,
    pass_at_ks: List[int],
    examples: List[Example],
    prompt_build_info: PromptBuildInfo,
    tokenizer: Any,
    client: SGLangRestClient,
    multiplex_self_check: Optional[MultiplexSelfCheckResult],
    run_started_at_unix: float,
) -> Dict[str, Any]:
    reasoning_prefix_tokens = int(args.hyperparam_reasoning_prefix_tokens)
    top_p_values = parse_float_values(args.hyperparam_top_p_values, label="hyperparam top-p")
    temperature_values = parse_float_values(
        args.hyperparam_temperature_values,
        label="hyperparam temperature",
    )
    condition_summaries: Dict[Tuple[str, str, float], Dict[str, Any]] = {}
    conditions: List[Tuple[str, str, float]] = []
    for phase in (HYPERPARAM_PHASE_THINKING, HYPERPARAM_PHASE_DISCRETE):
        conditions.extend(("top_p", phase, value) for value in top_p_values)
        conditions.extend(("temperature", phase, value) for value in temperature_values)

    for parameter_name, phase, value in conditions:
        apply_sampling_condition_to_args(
            args,
            phase=phase,
            parameter_name=parameter_name,
            parameter_value=value,
        )
        args.current_reasoning_prefix_tokens = reasoning_prefix_tokens
        condition_dir = hyperparam_condition_dir(
            output_dir,
            parameter_name=parameter_name,
            phase=phase,
            value=value,
        )
        condition_summaries[(parameter_name, phase, value)] = run_fixed_prefix_experiment(
            args=args,
            output_dir=condition_dir,
            runtime_config=runtime_config,
            scheduler_config=scheduler_config,
            pass_at_ks=pass_at_ks,
            examples=examples,
            methods=["baseline_independent", "shared_trace_branch_after_prefix"],
            prompt_build_info=prompt_build_info,
            tokenizer=tokenizer,
            client=client,
            multiplex_self_check=multiplex_self_check,
            reasoning_prefix_tokens=reasoning_prefix_tokens,
        )
        write_hyperparam_summary(
            output_dir,
            condition_summaries=condition_summaries,
            pass_at_ks=pass_at_ks,
            max_k=args.max_k,
            run_timing=build_run_timing(run_started_at_unix, time.time()),
        )
    apply_sampling_condition_to_args(args)
    write_hyperparam_summary(
        output_dir,
        condition_summaries=condition_summaries,
        pass_at_ks=pass_at_ks,
        max_k=args.max_k,
        run_timing=build_run_timing(run_started_at_unix, time.time()),
    )
    return {"conditions": condition_summaries}


def main() -> None:
    run_started_at_unix = time.time()
    args = parse_args()
    if args.compact_jsonl:
        os.environ[COMPACT_JSONL_ENV] = "1"
    apply_sampling_condition_to_args(args)
    if args.experiment_mode == EXPERIMENT_MODE_HYPERPARAM_SWEEP:
        args.max_k = 8
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
    run_standard_generation = (
        args.experiment_mode == EXPERIMENT_MODE_PASSK_SWEEP
        and "standard_generation_independent" in methods
    )
    pass_at_ks = build_pass_at_k_values(args.max_k)
    reasoning_prefix_token_values = parse_reasoning_prefix_token_values(
        args.reasoning_prefix_token_values
    )
    runtime_config = resolve_runtime_config(args)
    scheduler_config = resolve_scheduler_config(args, runtime_config)
    args.effective_dp_size = runtime_config.effective_dp_size
    args.effective_tp_size = runtime_config.effective_tp_size
    args.effective_request_batch_size = runtime_config.request_batch_size
    log_runtime_config(logger, runtime_config)
    log_scheduler_config(logger, scheduler_config)
    event_logger.log(
        "setup_runtime_resolved",
        **runtime_config_payload(runtime_config),
    )
    event_logger.log(
        "setup_scheduler_resolved",
        **asdict(scheduler_config),
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    examples, prompt_build_info = load_examples(tokenizer, args.benchmark, args.max_prompts)
    prompt_indices = parse_nonnegative_int_values(args.prompt_indices, label="prompt index")
    if prompt_indices:
        prompt_index_set = set(prompt_indices)
        examples = [example for example in examples if example.prompt_index in prompt_index_set]
        found_indices = {example.prompt_index for example in examples}
        missing_indices = [index for index in prompt_indices if index not in found_indices]
        if missing_indices:
            raise ValueError(
                "Requested prompt indices were not loaded by --max-prompts: "
                + ",".join(str(index) for index in missing_indices)
            )
        logger.info("[setup] Restricted run to prompt indices: %s", prompt_indices)
        event_logger.log("setup_prompt_indices_restricted", prompt_indices=prompt_indices)
    write_selection_manifest(output_dir, args, examples)
    runtime_probe_metadata: Optional[Dict[str, Any]] = None
    write_manifest(
        output_dir,
        args,
        runtime_config,
        scheduler_config,
        pass_at_ks,
        examples,
        methods,
        prompt_build_info,
        run_status="initializing",
        run_timing=build_run_timing(run_started_at_unix),
        reasoning_prefix_token_values=reasoning_prefix_token_values,
    )

    server_handle: Optional[SGLangServerHandle] = None
    client: Optional[SGLangRestClient] = None

    multiplex_self_check: Optional[MultiplexSelfCheckResult] = None
    prefix_summaries: Dict[int, Dict[str, Any]] = {}
    standard_summary: Optional[Dict[str, Any]] = None
    if args.resume and args.experiment_mode == EXPERIMENT_MODE_PASSK_SWEEP:
        prefix_summaries, standard_summary = load_existing_sweep_summaries(
            output_dir,
            pass_at_ks=pass_at_ks,
            max_k=args.max_k,
        )
    try:
        runtime_config, runtime_probe_metadata = autotune_runtime_config(
            args=args,
            runtime_config=runtime_config,
            tokenizer=tokenizer,
            logger=logger,
            event_logger=event_logger,
        )
        args.effective_dp_size = runtime_config.effective_dp_size
        args.effective_tp_size = runtime_config.effective_tp_size
        args.effective_request_batch_size = runtime_config.request_batch_size
        log_runtime_config(logger, runtime_config)
        event_logger.log(
            "setup_runtime_selected",
            selected_runtime_config=runtime_config_payload(runtime_config),
            runtime_probe=runtime_probe_metadata,
        )
        write_manifest(
            output_dir,
            args,
            runtime_config,
            scheduler_config,
            pass_at_ks,
            examples,
            methods,
            prompt_build_info,
            run_status="initializing",
            run_timing=build_run_timing(run_started_at_unix),
            reasoning_prefix_token_values=reasoning_prefix_token_values,
            runtime_probe_metadata=runtime_probe_metadata,
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
        client = make_rest_client(
            host=args.host,
            port=args.port,
            api_key=args.api_key,
            timeout=args.server_timeout_seconds,
        )
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
            scheduler_config,
            pass_at_ks,
            examples,
            methods,
            prompt_build_info,
            multiplex_self_check=multiplex_self_check,
            run_status="running",
            run_timing=build_run_timing(run_started_at_unix),
            reasoning_prefix_token_values=reasoning_prefix_token_values,
            runtime_probe_metadata=runtime_probe_metadata,
        )
        if not multiplex_self_check.success:
            raise RuntimeError(multiplex_self_check.message)

        if args.experiment_mode == EXPERIMENT_MODE_BRANCH_ABLATION:
            experiment_summary = run_branch_ablation_experiment(
                args=args,
                output_dir=output_dir,
                runtime_config=runtime_config,
                scheduler_config=scheduler_config,
                pass_at_ks=pass_at_ks,
                examples=examples,
                prompt_build_info=prompt_build_info,
                tokenizer=tokenizer,
                client=client,
                multiplex_self_check=multiplex_self_check,
                run_started_at_unix=run_started_at_unix,
            )
            run_timing = build_run_timing(run_started_at_unix, time.time())
            write_manifest(
                output_dir,
                args,
                runtime_config,
                scheduler_config,
                pass_at_ks,
                examples,
                methods,
                prompt_build_info,
                multiplex_self_check=multiplex_self_check,
                summary_json={"branch_ablation": experiment_summary},
                run_status="completed",
                run_timing=run_timing,
                reasoning_prefix_token_values=[int(args.branch_ablation_reasoning_prefix_tokens)],
                runtime_probe_metadata=runtime_probe_metadata,
            )
            logger.info(
                "[setup] Finished branch ablation in %s (%.1fs). Summary written to %s.",
                run_timing.get("wall_clock_hms"),
                float(run_timing.get("wall_clock_seconds", 0.0)),
                output_dir / "summary_ablation.md",
            )
            event_logger.log(
                "run_complete",
                output_dir=str(output_dir),
                experiment_mode=args.experiment_mode,
                wall_clock_seconds=run_timing.get("wall_clock_seconds"),
                wall_clock_hms=run_timing.get("wall_clock_hms"),
            )
            return

        if args.experiment_mode == EXPERIMENT_MODE_ADAPTIVE_ABLATION:
            experiment_summary = run_adaptive_ablation_experiment(
                args=args,
                output_dir=output_dir,
                runtime_config=runtime_config,
                scheduler_config=scheduler_config,
                pass_at_ks=pass_at_ks,
                examples=examples,
                prompt_build_info=prompt_build_info,
                tokenizer=tokenizer,
                client=client,
                multiplex_self_check=multiplex_self_check,
                run_started_at_unix=run_started_at_unix,
            )
            run_timing = build_run_timing(run_started_at_unix, time.time())
            write_manifest(
                output_dir,
                args,
                runtime_config,
                scheduler_config,
                pass_at_ks,
                examples,
                methods,
                prompt_build_info,
                multiplex_self_check=multiplex_self_check,
                summary_json={"adaptive_ablation": experiment_summary},
                run_status="completed",
                run_timing=run_timing,
                reasoning_prefix_token_values=[int(args.branch_ablation_reasoning_prefix_tokens)],
                runtime_probe_metadata=runtime_probe_metadata,
            )
            logger.info(
                "[setup] Finished adaptive ablation in %s (%.1fs). Summary written to %s.",
                run_timing.get("wall_clock_hms"),
                float(run_timing.get("wall_clock_seconds", 0.0)),
                output_dir / "summary_adaptive_ablation.md",
            )
            event_logger.log(
                "run_complete",
                output_dir=str(output_dir),
                experiment_mode=args.experiment_mode,
                wall_clock_seconds=run_timing.get("wall_clock_seconds"),
                wall_clock_hms=run_timing.get("wall_clock_hms"),
            )
            return

        if args.experiment_mode == EXPERIMENT_MODE_MEMORY_MATCH_TOPUP:
            experiment_summary = run_memory_match_topup_experiment(
                args=args,
                output_dir=output_dir,
                runtime_config=runtime_config,
                scheduler_config=scheduler_config,
                pass_at_ks=pass_at_ks,
                examples=examples,
                prompt_build_info=prompt_build_info,
                client=client,
                multiplex_self_check=multiplex_self_check,
                run_started_at_unix=run_started_at_unix,
            )
            run_timing = build_run_timing(run_started_at_unix, time.time())
            write_manifest(
                output_dir,
                args,
                runtime_config,
                scheduler_config,
                pass_at_ks,
                examples,
                methods,
                prompt_build_info,
                multiplex_self_check=multiplex_self_check,
                summary_json={"memory_match_topup": experiment_summary},
                run_status="completed",
                run_timing=run_timing,
                reasoning_prefix_token_values=[int(args.branch_ablation_reasoning_prefix_tokens)],
                runtime_probe_metadata=runtime_probe_metadata,
            )
            logger.info(
                "[setup] Finished memory-match top-up in %s (%.1fs). Summary written to %s.",
                run_timing.get("wall_clock_hms"),
                float(run_timing.get("wall_clock_seconds", 0.0)),
                output_dir / "summary_memory_match.md",
            )
            event_logger.log(
                "run_complete",
                output_dir=str(output_dir),
                experiment_mode=args.experiment_mode,
                wall_clock_seconds=run_timing.get("wall_clock_seconds"),
                wall_clock_hms=run_timing.get("wall_clock_hms"),
            )
            return

        if args.experiment_mode == EXPERIMENT_MODE_MIXED_MEMORY_MATCH_TOPUP:
            experiment_summary = run_mixed_memory_match_topup_experiment(
                args=args,
                output_dir=output_dir,
                runtime_config=runtime_config,
                scheduler_config=scheduler_config,
                pass_at_ks=pass_at_ks,
                examples=examples,
                prompt_build_info=prompt_build_info,
                client=client,
                multiplex_self_check=multiplex_self_check,
                run_started_at_unix=run_started_at_unix,
            )
            run_timing = build_run_timing(run_started_at_unix, time.time())
            write_manifest(
                output_dir,
                args,
                runtime_config,
                scheduler_config,
                pass_at_ks,
                examples,
                methods,
                prompt_build_info,
                multiplex_self_check=multiplex_self_check,
                summary_json={"mixed_memory_match_topup": experiment_summary},
                run_status="completed",
                run_timing=run_timing,
                reasoning_prefix_token_values=[int(args.branch_ablation_reasoning_prefix_tokens)],
                runtime_probe_metadata=runtime_probe_metadata,
            )
            logger.info(
                "[setup] Finished mixed memory-match top-up in %s (%.1fs). Summary written to %s.",
                run_timing.get("wall_clock_hms"),
                float(run_timing.get("wall_clock_seconds", 0.0)),
                output_dir / "summary_mixed_memory_match.md",
            )
            event_logger.log(
                "run_complete",
                output_dir=str(output_dir),
                experiment_mode=args.experiment_mode,
                wall_clock_seconds=run_timing.get("wall_clock_seconds"),
                wall_clock_hms=run_timing.get("wall_clock_hms"),
            )
            return

        if args.experiment_mode == EXPERIMENT_MODE_HYPERPARAM_SWEEP:
            experiment_summary = run_hyperparam_experiment(
                args=args,
                output_dir=output_dir,
                runtime_config=runtime_config,
                scheduler_config=scheduler_config,
                pass_at_ks=pass_at_ks,
                examples=examples,
                prompt_build_info=prompt_build_info,
                tokenizer=tokenizer,
                client=client,
                multiplex_self_check=multiplex_self_check,
                run_started_at_unix=run_started_at_unix,
            )
            run_timing = build_run_timing(run_started_at_unix, time.time())
            write_manifest(
                output_dir,
                args,
                runtime_config,
                scheduler_config,
                pass_at_ks,
                examples,
                methods,
                prompt_build_info,
                multiplex_self_check=multiplex_self_check,
                summary_json={"hyperparam_sweep": experiment_summary},
                run_status="completed",
                run_timing=run_timing,
                reasoning_prefix_token_values=[int(args.hyperparam_reasoning_prefix_tokens)],
                runtime_probe_metadata=runtime_probe_metadata,
            )
            logger.info(
                "[setup] Finished hyperparameter sweep in %s (%.1fs). Summary written to %s.",
                run_timing.get("wall_clock_hms"),
                float(run_timing.get("wall_clock_seconds", 0.0)),
                output_dir / "summary_hyperparam.md",
            )
            event_logger.log(
                "run_complete",
                output_dir=str(output_dir),
                experiment_mode=args.experiment_mode,
                wall_clock_seconds=run_timing.get("wall_clock_seconds"),
                wall_clock_hms=run_timing.get("wall_clock_hms"),
            )
            return

        if run_standard_generation:
            standard_dir = standard_generation_output_dir(output_dir)
            logger.info("[setup] Starting standard-generation baseline in %s.", standard_dir)
            standard_summary = run_standard_generation_experiment(
                args=args,
                output_dir=standard_dir,
                runtime_config=runtime_config,
                scheduler_config=scheduler_config,
                pass_at_ks=pass_at_ks,
                examples=examples,
                prompt_build_info=prompt_build_info,
                client=client,
            )
            prefix_summaries, standard_summary = write_prefix_sweep_summary_locked(
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
                    scheduler_config=scheduler_config,
                    pass_at_ks=pass_at_ks,
                    examples=examples,
                    methods=fixed_prefix_methods,
                    prompt_build_info=prompt_build_info,
                    tokenizer=tokenizer,
                    client=client,
                    multiplex_self_check=multiplex_self_check,
                    reasoning_prefix_tokens=reasoning_prefix_tokens,
                )
                prefix_summaries, standard_summary = write_prefix_sweep_summary_locked(
                    output_dir,
                    prefix_summaries,
                    standard_summary,
                    pass_at_ks,
                    args.max_k,
                    run_timing=build_run_timing(run_started_at_unix, time.time()),
                )

        run_timing = build_run_timing(run_started_at_unix, time.time())
        prefix_summaries, standard_summary = write_prefix_sweep_summary_locked(
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
            scheduler_config,
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
            runtime_probe_metadata=runtime_probe_metadata,
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
            scheduler_config,
            pass_at_ks,
            examples,
            methods,
            prompt_build_info,
            multiplex_self_check=multiplex_self_check,
            run_status="aborted",
            run_failure=run_failure,
            run_timing=run_timing,
            reasoning_prefix_token_values=reasoning_prefix_token_values,
            runtime_probe_metadata=runtime_probe_metadata,
        )
        logger.info(
            "[setup] Run aborted cleanly after %s. Manifest updated at %s.",
            run_timing.get("wall_clock_hms"),
            output_dir / "manifest.json",
        )
        raise
    finally:
        if client is not None:
            client.close()
        if server_handle is not None:
            server_handle.stop()


if __name__ == "__main__":
    main()
