#!/usr/bin/env python3
"""Minimal soft-thinking generation smokes.

This intentionally avoids the PPO trainer.  It answers one question at a time:
does the generation backend return a soft-thinking prefix, and can a 4x4
shared-prefix branch be continued synchronously?
"""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import json
import multiprocessing as mp
import os
import queue
import signal
import sys
import time
from dataclasses import dataclass
from typing import Any


DEFAULT_PROMPT = (
    "Solve the problem. A rectangle has perimeter 30 and integer side lengths. "
    "What is the largest possible area?"
)


def truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["sglang-vanilla", "hf-branch"], required=True)
    parser.add_argument("--model", default=os.environ.get("MODEL_PATH", "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"))
    parser.add_argument("--prompt", action="append", default=None)
    parser.add_argument("--multiplex-width", type=int, default=int(os.environ.get("MULTIPLEX_WIDTH", "3")))
    parser.add_argument("--thinking-tokens", type=int, default=int(os.environ.get("SMOKE_THINKING_TOKENS", "8")))
    parser.add_argument("--continuation-tokens", type=int, default=int(os.environ.get("SMOKE_CONTINUATION_TOKENS", "8")))
    parser.add_argument("--traces", type=int, default=int(os.environ.get("SMOKE_TRACES", "4")))
    parser.add_argument("--answers-per-trace", type=int, default=int(os.environ.get("SMOKE_ANSWERS_PER_TRACE", "4")))
    parser.add_argument("--timeout-seconds", type=int, default=int(os.environ.get("SOFT_SMOKE_TIMEOUT_SECONDS", "240")))
    parser.add_argument(
        "--generation-timeout-seconds",
        type=int,
        default=int(os.environ.get("SOFT_SMOKE_GENERATION_TIMEOUT_SECONDS", "180")),
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SOFT_SMOKE_SEED", "1234")))
    parser.add_argument("--dtype", default=os.environ.get("SOFT_SMOKE_DTYPE", "bfloat16"))
    parser.add_argument("--gpu-mem-util", type=float, default=float(os.environ.get("GPU_MEM_UTIL", "0.50")))
    parser.add_argument("--attention-backend", default=os.environ.get("SGLANG_ATTENTION_BACKEND", "flashinfer"))
    parser.add_argument("--sampling-backend", default=os.environ.get("SGLANG_SAMPLING_BACKEND", "pytorch"))
    parser.add_argument("--attn-implementation", default=os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"))
    parser.add_argument("--max-running-requests", type=int, default=int(os.environ.get("ROLLOUT_MAX_NUM_SEQS", "8")))
    return parser.parse_args()


def run_with_process_timeout(args: argparse.Namespace) -> dict[str, Any]:
    ctx = mp.get_context("spawn")
    result_queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(target=_child_main, args=(args, result_queue), daemon=False)
    proc.start()
    proc.join(args.timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(20)
        if proc.is_alive():
            os.kill(proc.pid, signal.SIGKILL)
            proc.join(10)
        return {
            "ok": False,
            "backend": args.backend,
            "error": f"hard timeout after {args.timeout_seconds}s",
            "exitcode": proc.exitcode,
        }
    try:
        result = result_queue.get_nowait()
    except queue.Empty:
        result = {
            "ok": proc.exitcode == 0,
            "backend": args.backend,
            "error": "child exited without returning a result",
            "exitcode": proc.exitcode,
        }
    result["exitcode"] = proc.exitcode
    return result


def _child_main(args: argparse.Namespace, result_queue: mp.Queue) -> None:
    faulthandler.enable()
    faulthandler.dump_traceback_later(args.generation_timeout_seconds, repeat=False)
    os.environ.setdefault("FORCE_THINK_END_AT_LENGTH", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        if args.backend == "sglang-vanilla":
            result = _run_sglang_vanilla(args)
        elif args.backend == "hf-branch":
            result = _run_hf_branch(args)
        else:
            raise ValueError(f"unknown backend: {args.backend}")
        result["ok"] = True
        result["backend"] = args.backend
        result_queue.put(result)
    except BaseException as exc:
        faulthandler.dump_traceback(file=sys.stderr)
        result_queue.put(
            {
                "ok": False,
                "backend": args.backend,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        raise
    finally:
        faulthandler.cancel_dump_traceback_later()


def _torch_dtype(name: str):
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name.lower()]


async def _await_with_timeout(awaitable, timeout_seconds: int, label: str):
    started = time.monotonic()
    try:
        return await asyncio.wait_for(awaitable, timeout=timeout_seconds)
    except asyncio.TimeoutError as exc:
        elapsed = time.monotonic() - started
        raise TimeoutError(f"{label} did not return within {timeout_seconds}s; elapsed={elapsed:.1f}s") from exc


def _run_sglang_vanilla(args: argparse.Namespace) -> dict[str, Any]:
    from sglang.srt.entrypoints.engine import Engine

    prompts = args.prompt or [DEFAULT_PROMPT]
    port_base = int(os.environ.get("SGLANG_PORT_BASE", str(47000 + (os.getpid() % 10000))))
    engine_kwargs = {
        "model_path": args.model,
        "dtype": args.dtype,
        "mem_fraction_static": args.gpu_mem_util,
        "enable_memory_saver": False,
        "enable_soft_thinking": True,
        "max_topk": args.multiplex_width,
        "used_topk": args.multiplex_width,
        "enable_entropy_mask": False,
        "entropy_mask_threshold": 0.0,
        "early_stopping_entropy_threshold": -1.0,
        "early_stopping_length_threshold": args.thinking_tokens,
        "think_end_str": "</think>",
        "dirichlet_alpha": 1.0,
        "enable_gumbel": False,
        "enable_max_topk": False,
        "gumbel_tau": 1.0,
        "enable_replacement": True,
        "enable_gumbel_after_thinking": False,
        "enable_unweighting": False,
        "after_thinking_temperature": 1.0,
        "after_thinking_top_p": 1.0,
        "after_thinking_top_k": -1,
        "after_thinking_min_p": 0.0,
        "disable_overlap_schedule": True,
        "disable_cuda_graph": True,
        "sampling_backend": args.sampling_backend,
        "grammar_backend": "none",
        "watchdog_timeout": max(args.generation_timeout_seconds, 60),
        "max_running_requests": args.max_running_requests,
        "port": port_base,
        "log_level": "info",
    }
    if args.attention_backend:
        engine_kwargs["attention_backend"] = args.attention_backend
    print(f"[soft-smoke] launching SGLang engine kwargs={_jsonable(engine_kwargs)}", flush=True)
    engine = Engine(**engine_kwargs)
    sampling_params = {
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "max_new_tokens": args.thinking_tokens + 1,
        "ignore_eos": True,
        "think_end_str": "</think>",
        "early_stopping_entropy_threshold": -1.0,
        "early_stopping_length_threshold": args.thinking_tokens,
        "after_thinking_temperature": 1.0,
        "after_thinking_top_p": 1.0,
        "after_thinking_top_k": -1,
        "after_thinking_min_p": 0.0,
        "max_topk": args.multiplex_width,
        "used_topk": args.multiplex_width,
        "enable_entropy_mask": False,
        "entropy_mask_threshold": 0.0,
        "enable_gumbel": False,
        "enable_max_topk": False,
        "gumbel_tau": 1.0,
        "enable_replacement": True,
        "enable_gumbel_after_thinking": False,
        "enable_unweighting": False,
    }
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        output = loop.run_until_complete(
            _await_with_timeout(
                engine.async_generate(
                    prompt=prompts,
                    sampling_params=sampling_params,
                    return_logprob=True,
                ),
                args.generation_timeout_seconds,
                "sglang soft-thinking generate",
            )
        )
    finally:
        try:
            engine.shutdown()
        except BaseException as exc:
            print(f"[soft-smoke] engine shutdown warning: {exc}", flush=True)

    rows = output if isinstance(output, list) else [output]
    summaries = []
    for row in rows:
        meta = row.get("meta_info", {})
        token_ids = meta.get("output_token_ids") or row.get("output_ids") or []
        topk_probs = meta.get("output_topk_probs_list")
        topk_indices = meta.get("output_topk_indices_list")
        if not topk_probs or not topk_indices:
            raise RuntimeError(f"SGLang returned no soft-thinking top-k tensors; meta_keys={sorted(meta.keys())}")
        if len(topk_probs[0]) != args.multiplex_width:
            raise RuntimeError(f"expected top-k width {args.multiplex_width}, got {len(topk_probs[0])}")
        summaries.append(
            {
                "token_count": len(token_ids),
                "topk_steps": len(topk_probs),
                "topk_width": len(topk_probs[0]),
                "finish_reason": meta.get("finish_reason"),
                "text_preview": (row.get("text") or "")[:160],
            }
        )
    return {
        "model": args.model,
        "prompt_count": len(prompts),
        "multiplex_width": args.multiplex_width,
        "thinking_tokens": args.thinking_tokens,
        "rows": summaries,
    }


@dataclass
class SoftPrefix:
    prompt_input_ids: Any
    prompt_attention_mask: Any
    topk_probs: Any
    topk_indices: Any
    primary_tokens: Any


def _run_hf_branch(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(args.seed)
    if not torch.cuda.is_available():
        raise RuntimeError("hf-branch smoke requires CUDA")
    prompts = args.prompt or [DEFAULT_PROMPT]
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=_torch_dtype(args.dtype),
        attn_implementation=args.attn_implementation,
        trust_remote_code=False,
    ).to("cuda")
    model.eval()

    with torch.inference_mode():
        prefix = _hf_soft_prefix(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            traces=args.traces,
            thinking_tokens=args.thinking_tokens,
            width=args.multiplex_width,
        )
        branch_tokens = _hf_continue_from_soft_prefix(
            model=model,
            tokenizer=tokenizer,
            prefix=prefix,
            answers_per_trace=args.answers_per_trace,
            continuation_tokens=args.continuation_tokens,
        )

    expected_rows = len(prompts) * args.traces * args.answers_per_trace
    if branch_tokens.shape != (expected_rows, args.continuation_tokens):
        raise RuntimeError(f"unexpected branch token shape {tuple(branch_tokens.shape)} expected {(expected_rows, args.continuation_tokens)}")
    decoded = tokenizer.batch_decode(branch_tokens[: min(4, expected_rows)], skip_special_tokens=False)
    return {
        "model": args.model,
        "prompt_count": len(prompts),
        "multiplex_width": args.multiplex_width,
        "traces": args.traces,
        "answers_per_trace": args.answers_per_trace,
        "rollout_n": args.traces * args.answers_per_trace,
        "thinking_tokens": args.thinking_tokens,
        "prefix_length": args.thinking_tokens + 1,
        "continuation_tokens": args.continuation_tokens,
        "branch_shape": list(branch_tokens.shape),
        "topk_shape": list(prefix.topk_probs.shape),
        "decoded_preview": [text[:160] for text in decoded],
    }


def _hf_soft_prefix(model, tokenizer, prompts: list[str], traces: int, thinking_tokens: int, width: int) -> SoftPrefix:
    import torch

    expanded_prompts = [prompt for prompt in prompts for _ in range(traces)]
    encoded = tokenizer(expanded_prompts, return_tensors="pt", padding=True).to("cuda")
    out = model(**encoded, use_cache=True)
    logits = out.logits[:, -1, :]
    past = out.past_key_values
    attention_mask = encoded["attention_mask"]
    topk_probs_rows = []
    topk_indices_rows = []
    primary_rows = []

    for _ in range(thinking_tokens):
        probs = _renormalized_probs(logits, top_k=-1, top_p=1.0, min_p=0.0)
        topk_indices = torch.multinomial(probs, num_samples=width, replacement=True)
        topk_probs = torch.gather(probs, dim=-1, index=topk_indices)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        topk_probs_rows.append(topk_probs.to(torch.bfloat16))
        topk_indices_rows.append(topk_indices)
        primary_rows.append(topk_indices[:, 0])
        weighted_embed = _weighted_embeddings(model, topk_probs, topk_indices)
        attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device="cuda", dtype=attention_mask.dtype)], dim=1)
        out = model(inputs_embeds=weighted_embed, attention_mask=attention_mask, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values

    think_end_id = tokenizer.encode("</think>", add_special_tokens=False)[-1]
    batch = logits.shape[0]
    forced_indices = torch.zeros((batch, width), device="cuda", dtype=torch.long)
    forced_probs = torch.zeros((batch, width), device="cuda", dtype=torch.bfloat16)
    forced_indices[:, 0] = think_end_id
    forced_probs[:, 0] = 1.0
    topk_probs_rows.append(forced_probs)
    topk_indices_rows.append(forced_indices)
    primary_rows.append(forced_indices[:, 0])
    return SoftPrefix(
        prompt_input_ids=encoded["input_ids"],
        prompt_attention_mask=encoded["attention_mask"],
        topk_probs=torch.stack(topk_probs_rows, dim=1),
        topk_indices=torch.stack(topk_indices_rows, dim=1),
        primary_tokens=torch.stack(primary_rows, dim=1),
    )


def _hf_continue_from_soft_prefix(model, tokenizer, prefix: SoftPrefix, answers_per_trace: int, continuation_tokens: int):
    import torch

    prompt_embeds = model.get_input_embeddings()(prefix.prompt_input_ids)
    prefix_embeds = _weighted_prefix_embeddings(model, prefix.topk_probs.float(), prefix.topk_indices)
    full_embeds = torch.cat([prompt_embeds, prefix_embeds], dim=1)
    prefix_mask = torch.ones(
        (prefix.prompt_attention_mask.shape[0], prefix.topk_probs.shape[1]),
        device="cuda",
        dtype=prefix.prompt_attention_mask.dtype,
    )
    full_mask = torch.cat([prefix.prompt_attention_mask, prefix_mask], dim=1)
    full_embeds = full_embeds.repeat_interleave(answers_per_trace, dim=0)
    full_mask = full_mask.repeat_interleave(answers_per_trace, dim=0)

    out = model(inputs_embeds=full_embeds, attention_mask=full_mask, use_cache=True)
    logits = out.logits[:, -1, :]
    past = out.past_key_values
    attention_mask = full_mask
    generated = []
    for _ in range(continuation_tokens):
        probs = _renormalized_probs(logits, top_k=-1, top_p=1.0, min_p=0.0)
        token = torch.multinomial(probs, num_samples=1).squeeze(-1)
        generated.append(token)
        token_embed = model.get_input_embeddings()(token).unsqueeze(1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), device="cuda", dtype=attention_mask.dtype)],
            dim=1,
        )
        out = model(inputs_embeds=token_embed, attention_mask=attention_mask, past_key_values=past, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
    return torch.stack(generated, dim=1)


def _weighted_embeddings(model, topk_probs, topk_indices):
    embeds = model.get_input_embeddings()(topk_indices)
    weights = topk_probs.to(dtype=embeds.dtype)
    return (embeds * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)


def _weighted_prefix_embeddings(model, topk_probs, topk_indices):
    embeds = model.get_input_embeddings()(topk_indices)
    weights = topk_probs.to(dtype=embeds.dtype)
    return (embeds * weights.unsqueeze(-1)).sum(dim=2)


def _renormalized_probs(logits, top_k: int, top_p: float, min_p: float):
    import torch
    import torch.nn.functional as F

    probs = F.softmax(logits.float(), dim=-1)
    if top_k is not None and top_k > 0 and top_k < probs.shape[-1]:
        kth = torch.topk(probs, k=top_k, dim=-1).values[:, -1:]
        probs = probs.masked_fill(probs < kth, 0.0)
    if top_p is not None and top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[:, 1:] = remove[:, :-1].clone()
        remove[:, 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        probs = torch.zeros_like(probs).scatter(-1, sorted_idx, sorted_probs)
    if min_p is not None and min_p > 0.0:
        threshold = probs.max(dim=-1, keepdim=True).values * min_p
        probs = probs.masked_fill(probs < threshold, 0.0)
    denom = probs.sum(dim=-1, keepdim=True)
    if torch.any(denom <= 0):
        raise RuntimeError("sampling distribution collapsed to zero probability")
    return probs / denom


def _jsonable(obj: Any) -> Any:
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        if isinstance(obj, dict):
            return {key: _jsonable(value) for key, value in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(value) for value in obj]
        return str(obj)


def main() -> int:
    args = parse_args()
    result = run_with_process_timeout(args)
    print("[soft-smoke-result] " + json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
