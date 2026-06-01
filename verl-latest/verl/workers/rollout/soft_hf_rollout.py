# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Synchronous PyTorch/HF rollout for soft-thinking experiments.

This backend intentionally bypasses SGLang.  It is slower, but it preserves the
soft-thinking contract by sampling weighted top-k rows, feeding weighted
embeddings back into the model, and returning the same top-k tensors consumed by
the actor loss.
"""

from __future__ import annotations

import contextlib
import faulthandler
import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any, Generator

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device
from verl.utils.torch_functional import get_response_mask
from verl.workers.rollout.base import BaseRollout


@dataclass
class _SoftPrefix:
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor
    topk_probs: torch.Tensor
    topk_indices: torch.Tensor
    primary_tokens: torch.Tensor


@contextlib.contextmanager
def _generation_timeout(seconds: int, label: str):
    """Dump stacks and raise if Python regains control after a long generation."""
    if seconds <= 0:
        yield
        return

    def _raise_timeout(signum, frame):  # noqa: ARG001
        raise TimeoutError(f"{label} exceeded {seconds}s")

    use_alarm = threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGALRM")
    previous_handler = None
    try:
        faulthandler.dump_traceback_later(seconds, repeat=False)
        if use_alarm:
            previous_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _raise_timeout)
            signal.setitimer(signal.ITIMER_REAL, seconds)
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
        if use_alarm:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)


class SoftHFRollout(BaseRollout):
    """HF/PyTorch soft-thinking rollout using the actor module directly."""

    def __init__(self, config, model_config, device_mesh, module):
        super().__init__(config=config, model_config=model_config, device_mesh=device_mesh)
        self.module = module
        self.tokenizer = model_config.tokenizer
        self.pad_token_id = self._resolve_pad_token_id()
        self.eos_token_id = self._resolve_eos_token_id()
        self.think_end_id = self._resolve_think_end_id()
        self.timeout_seconds = int(os.environ.get("PYTORCH_ROLLOUT_TIMEOUT_SECONDS", "900"))
        self._debug = os.environ.get("PYTORCH_ROLLOUT_DEBUG", "").lower() in {"1", "true", "yes", "on"}

    async def resume(self, tags: list[str]):  # noqa: ARG002
        return None

    async def update_weights(self, weights: Generator[tuple[str, torch.Tensor], None, None], **kwargs):  # noqa: ARG002
        return None

    async def release(self):
        get_torch_device().empty_cache()
        return None

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch["input_ids"].shape[0]
        if batch_size == 0:
            raise ValueError("SoftHFRollout received an empty prompt batch")

        started = time.monotonic()
        if self._debug:
            print(
                "[soft_hf_rollout] generate begin "
                f"batch={batch_size} response_length={self.config.response_length} "
                f"branch={self.config.get('branch_rollout', False)}",
                flush=True,
            )
        was_training = self.module.training
        self.module.eval()
        try:
            with _generation_timeout(self.timeout_seconds, "SoftHFRollout.generate_sequences"):
                if self._use_soft_rollout(prompts):
                    output = self._generate_soft(prompts)
                else:
                    output = self._generate_discrete(prompts)
        finally:
            if was_training:
                self.module.train()
            get_torch_device().empty_cache()

        if self._debug:
            elapsed = time.monotonic() - started
            print(f"[soft_hf_rollout] generate end elapsed={elapsed:.1f}s", flush=True)
        return output

    def _use_soft_rollout(self, prompts: DataProto) -> bool:
        return (
            bool(self.config.get("enable_soft_thinking", False))
            and bool(prompts.meta_info.get("do_sample", self.config.do_sample))
            and not bool(prompts.meta_info.get("validate", False))
        )

    def _generate_soft(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        batch_size = idx.shape[0]
        response_length = int(self.config.response_length)
        width = int(self.config.get("used_topk", self.config.get("max_topk", 3)))
        if width <= 0:
            raise ValueError(f"soft rollout requires positive used_topk, got {width}")

        if bool(self.config.get("branch_rollout", False)):
            response, topk_probs, topk_indices = self._generate_shared_branch(idx, attention_mask, width)
        else:
            response, topk_probs, topk_indices = self._generate_independent_soft(idx, attention_mask, width)

        if response.shape != (batch_size, response_length):
            raise RuntimeError(
                f"soft rollout produced response shape {tuple(response.shape)}, expected {(batch_size, response_length)}"
            )
        if topk_probs.shape != (batch_size, response_length, width):
            raise RuntimeError(
                "soft rollout produced topk_probs shape "
                f"{tuple(topk_probs.shape)}, expected {(batch_size, response_length, width)}"
            )

        return self._build_output(prompts, idx, attention_mask, position_ids, response, topk_probs, topk_indices)

    def _generate_shared_branch(
        self, idx: torch.Tensor, attention_mask: torch.Tensor, width: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        traces = int(self.config.get("branch_rollout_thinking_traces", 4))
        answers = int(self.config.get("branch_rollout_answers_per_trace", 4))
        rollout_n = int(self.config.get("n", traces * answers))
        if traces <= 0 or answers <= 0:
            raise ValueError("branch rollout requires positive traces and answers")
        if rollout_n != traces * answers:
            raise ValueError(f"branch rollout expected n={traces * answers}, got {rollout_n}")
        if idx.shape[0] % rollout_n != 0:
            raise ValueError(f"local batch {idx.shape[0]} is not divisible by rollout_n={rollout_n}")

        thinking_tokens = self._thinking_tokens()
        continuation_tokens = self._continuation_tokens(thinking_tokens)
        prompt_count = idx.shape[0] // rollout_n
        base_rows = torch.arange(prompt_count, device=idx.device) * rollout_n
        base_ids = idx.index_select(0, base_rows)
        base_mask = attention_mask.index_select(0, base_rows)

        prefix = self._soft_prefix_from_prompt_ids(
            base_ids=base_ids,
            base_attention_mask=base_mask,
            repeats=traces,
            thinking_tokens=thinking_tokens,
            width=width,
        )
        continuation = self._continue_from_soft_prefix(prefix, answers, continuation_tokens)
        return self._stitch_branch_outputs(prefix, continuation, prompt_count, traces, answers, continuation_tokens, width)

    def _generate_independent_soft(
        self, idx: torch.Tensor, attention_mask: torch.Tensor, width: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        thinking_tokens = self._thinking_tokens()
        continuation_tokens = self._continuation_tokens(thinking_tokens)
        prefix = self._soft_prefix_from_prompt_ids(
            base_ids=idx,
            base_attention_mask=attention_mask,
            repeats=1,
            thinking_tokens=thinking_tokens,
            width=width,
        )
        continuation = self._continue_from_soft_prefix(prefix, 1, continuation_tokens)
        response = torch.cat([prefix.primary_tokens, continuation], dim=1)
        cont_probs, cont_indices = self._one_hot_topk(continuation, width)
        topk_probs = torch.cat([prefix.topk_probs, cont_probs], dim=1)
        topk_indices = torch.cat([prefix.topk_indices, cont_indices], dim=1)
        return response, topk_probs, topk_indices

    def _soft_prefix_from_prompt_ids(
        self,
        base_ids: torch.Tensor,
        base_attention_mask: torch.Tensor,
        repeats: int,
        thinking_tokens: int,
        width: int,
    ) -> _SoftPrefix:
        prompt_ids = base_ids.repeat_interleave(repeats, dim=0)
        prompt_mask = base_attention_mask.repeat_interleave(repeats, dim=0)
        out = self.module(input_ids=prompt_ids, attention_mask=prompt_mask, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        running_mask = prompt_mask
        topk_probs_rows = []
        topk_indices_rows = []
        primary_rows = []

        for step in range(thinking_tokens):
            self._check_elapsed(step, thinking_tokens, "soft-prefix")
            probs = self._filtered_probs(
                logits,
                temperature=float(self.config.get("temperature", 1.0)),
                top_k=int(self.config.get("top_k", -1)),
                top_p=float(self.config.get("top_p", 1.0)),
                min_p=0.0,
            )
            topk_indices, topk_probs = self._sample_topk(probs, width)
            topk_probs_rows.append(topk_probs.to(torch.bfloat16))
            topk_indices_rows.append(topk_indices)
            primary_rows.append(topk_indices[:, 0])

            weighted_embed = self._weighted_embeddings(topk_probs, topk_indices)
            running_mask = self._append_attention(running_mask, active=None)
            out = self.module(
                inputs_embeds=weighted_embed,
                attention_mask=running_mask,
                past_key_values=past,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]
            past = out.past_key_values

        forced_indices = torch.full((prompt_ids.shape[0], width), self.pad_token_id, device=prompt_ids.device)
        forced_probs = torch.zeros((prompt_ids.shape[0], width), dtype=torch.bfloat16, device=prompt_ids.device)
        forced_indices[:, 0] = self.think_end_id
        forced_probs[:, 0] = 1.0
        topk_probs_rows.append(forced_probs)
        topk_indices_rows.append(forced_indices)
        primary_rows.append(forced_indices[:, 0])

        return _SoftPrefix(
            prompt_input_ids=prompt_ids,
            prompt_attention_mask=prompt_mask,
            topk_probs=torch.stack(topk_probs_rows, dim=1),
            topk_indices=torch.stack(topk_indices_rows, dim=1),
            primary_tokens=torch.stack(primary_rows, dim=1),
        )

    def _continue_from_soft_prefix(
        self, prefix: _SoftPrefix, answers_per_trace: int, continuation_tokens: int
    ) -> torch.Tensor:
        if continuation_tokens <= 0:
            return torch.empty((prefix.primary_tokens.shape[0] * answers_per_trace, 0), dtype=torch.long, device=prefix.primary_tokens.device)

        prompt_embeds = self._input_embeddings()(prefix.prompt_input_ids)
        prefix_embeds = self._weighted_prefix_embeddings(prefix.topk_probs.float(), prefix.topk_indices)
        full_embeds = torch.cat([prompt_embeds, prefix_embeds], dim=1)
        prefix_mask = torch.ones(
            (prefix.prompt_attention_mask.shape[0], prefix.topk_probs.shape[1]),
            device=prefix.prompt_attention_mask.device,
            dtype=prefix.prompt_attention_mask.dtype,
        )
        full_mask = torch.cat([prefix.prompt_attention_mask, prefix_mask], dim=1)
        full_embeds = full_embeds.repeat_interleave(answers_per_trace, dim=0)
        running_mask = full_mask.repeat_interleave(answers_per_trace, dim=0)

        out = self.module(inputs_embeds=full_embeds, attention_mask=running_mask, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        generated = []
        for step in range(continuation_tokens):
            self._check_elapsed(step, continuation_tokens, "continuation")
            probs = self._filtered_probs(
                logits,
                temperature=float(self.config.get("after_thinking_temperature", self.config.get("temperature", 1.0))),
                top_k=int(self.config.get("after_thinking_top_k", self.config.get("top_k", -1))),
                top_p=float(self.config.get("after_thinking_top_p", self.config.get("top_p", 1.0))),
                min_p=float(self.config.get("after_thinking_min_p", 0.0)),
            )
            token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            generated.append(token)
            token_embed = self._input_embeddings()(token).unsqueeze(1)
            running_mask = self._append_attention(running_mask, active=None)
            out = self.module(
                inputs_embeds=token_embed,
                attention_mask=running_mask,
                past_key_values=past,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]
            past = out.past_key_values
        return torch.stack(generated, dim=1)

    def _stitch_branch_outputs(
        self,
        prefix: _SoftPrefix,
        continuation: torch.Tensor,
        prompt_count: int,
        traces: int,
        answers: int,
        continuation_tokens: int,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        response_rows = []
        topk_prob_rows = []
        topk_index_rows = []
        for prompt_idx in range(prompt_count):
            for trace_idx in range(traces):
                prefix_row = prompt_idx * traces + trace_idx
                for answer_idx in range(answers):
                    cont_row = (prompt_idx * traces + trace_idx) * answers + answer_idx
                    cont = continuation[cont_row]
                    response_rows.append(torch.cat([prefix.primary_tokens[prefix_row], cont], dim=0))
                    cont_probs, cont_indices = self._one_hot_topk(cont.unsqueeze(0), width)
                    topk_prob_rows.append(torch.cat([prefix.topk_probs[prefix_row], cont_probs.squeeze(0)], dim=0))
                    topk_index_rows.append(torch.cat([prefix.topk_indices[prefix_row], cont_indices.squeeze(0)], dim=0))

        response = torch.stack(response_rows, dim=0)
        topk_probs = torch.stack(topk_prob_rows, dim=0)
        topk_indices = torch.stack(topk_index_rows, dim=0)
        expected_len = int(self.config.response_length)
        if continuation_tokens == 0 and response.shape[1] != expected_len:
            pad_len = expected_len - response.shape[1]
            if pad_len < 0:
                raise RuntimeError(f"response length {response.shape[1]} exceeds configured {expected_len}")
            response = F.pad(response, (0, pad_len), value=self.pad_token_id)
            topk_probs = F.pad(topk_probs, (0, 0, 0, pad_len), value=0.0)
            topk_indices = F.pad(topk_indices, (0, 0, 0, pad_len), value=self.pad_token_id)
        return response, topk_probs, topk_indices

    def _generate_discrete(self, prompts: DataProto) -> DataProto:
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        response_length = int(self.config.response_length)
        do_sample = bool(prompts.meta_info.get("do_sample", self.config.do_sample))
        out = self.module(input_ids=idx, attention_mask=attention_mask, use_cache=True)
        logits = out.logits[:, -1, :]
        past = out.past_key_values
        running_mask = attention_mask
        generated = []
        for step in range(response_length):
            self._check_elapsed(step, response_length, "discrete")
            if do_sample:
                probs = self._filtered_probs(
                    logits,
                    temperature=float(self.config.get("temperature", 1.0)),
                    top_k=int(self.config.get("top_k", -1)),
                    top_p=float(self.config.get("top_p", 1.0)),
                    min_p=0.0,
                )
                token = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                token = torch.argmax(logits, dim=-1)
            generated.append(token)
            token_embed = self._input_embeddings()(token).unsqueeze(1)
            running_mask = self._append_attention(running_mask, active=None)
            out = self.module(
                inputs_embeds=token_embed,
                attention_mask=running_mask,
                past_key_values=past,
                use_cache=True,
            )
            logits = out.logits[:, -1, :]
            past = out.past_key_values
        response = torch.stack(generated, dim=1)
        return self._build_output(prompts, idx, attention_mask, position_ids, response, None, None)

    def _build_output(
        self,
        prompts: DataProto,
        idx: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        response: torch.Tensor,
        topk_probs: torch.Tensor | None,
        topk_indices: torch.Tensor | None,
    ) -> DataProto:
        batch_size = response.shape[0]
        response_length = response.shape[1]
        seq = torch.cat([idx, response], dim=-1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, 3, -1)
        response_position_ids = position_ids[..., -1:] + delta_position_id
        full_position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response,
            eos_token=prompts.meta_info.get("eos_token_id", self.eos_token_id),
            dtype=attention_mask.dtype,
        )
        full_attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "attention_mask": full_attention_mask,
                "position_ids": full_position_ids,
            },
            batch_size=batch_size,
        )
        if topk_probs is not None:
            batch["soft_thinking_topk_probs"] = topk_probs
        if topk_indices is not None:
            batch["soft_thinking_topk_indices"] = topk_indices
        return DataProto(batch=batch, non_tensor_batch=prompts.non_tensor_batch)

    def _sample_topk(self, probs: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        replacement = bool(self.config.get("enable_replacement", True))
        topk_indices = torch.multinomial(probs, num_samples=width, replacement=replacement)
        topk_probs = torch.gather(probs, dim=-1, index=topk_indices)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-20)
        return topk_indices, topk_probs

    def _filtered_probs(
        self, logits: torch.Tensor, temperature: float, top_k: int, top_p: float, min_p: float
    ) -> torch.Tensor:
        if temperature <= 0:
            indices = torch.argmax(logits, dim=-1, keepdim=True)
            probs = torch.zeros_like(logits, dtype=torch.float32)
            probs.scatter_(1, indices, 1.0)
            return probs
        probs = F.softmax((logits / temperature).float(), dim=-1)
        if top_k is not None and 0 < top_k < probs.shape[-1]:
            kth = torch.topk(probs, k=top_k, dim=-1).values[:, -1:]
            probs = probs.masked_fill(probs < kth, 0.0)
        if top_p is not None and top_p < 1.0:
            sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            remove = cumulative > top_p
            remove[:, 1:] = remove[:, :-1].clone()
            remove[:, 0] = False
            scatter_remove = torch.zeros_like(remove).scatter(1, sorted_idx, remove)
            probs = probs.masked_fill(scatter_remove, 0.0)
        if min_p is not None and min_p > 0.0:
            probs = probs.masked_fill(probs < min_p, 0.0)
        return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-20)

    def _weighted_embeddings(self, topk_probs: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        embeds = self._input_embeddings()(topk_indices)
        weights = topk_probs.to(dtype=embeds.dtype)
        return (embeds * weights.unsqueeze(-1)).sum(dim=1, keepdim=True)

    def _weighted_prefix_embeddings(self, topk_probs: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
        embeds = self._input_embeddings()(topk_indices)
        weights = topk_probs.to(dtype=embeds.dtype)
        return (embeds * weights.unsqueeze(-1)).sum(dim=2)

    def _one_hot_topk(self, tokens: torch.Tensor, width: int) -> tuple[torch.Tensor, torch.Tensor]:
        probs = torch.zeros((*tokens.shape, width), dtype=torch.bfloat16, device=tokens.device)
        indices = torch.full((*tokens.shape, width), self.pad_token_id, dtype=torch.long, device=tokens.device)
        if tokens.numel() > 0:
            probs[..., 0] = 1.0
            indices[..., 0] = tokens
        return probs, indices

    def _append_attention(self, attention_mask: torch.Tensor, active: torch.Tensor | None) -> torch.Tensor:
        next_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
        if active is not None:
            next_mask = next_mask * active.to(dtype=attention_mask.dtype).unsqueeze(-1)
        return torch.cat([attention_mask, next_mask], dim=1)

    def _input_embeddings(self):
        module = getattr(self.module, "_fsdp_wrapped_module", self.module)
        return module.get_input_embeddings()

    def _thinking_tokens(self) -> int:
        thinking_tokens = int(
            self.config.get("branch_rollout_thinking_tokens", 0)
            or self.config.get("early_stopping_length_threshold", 0)
        )
        if thinking_tokens <= 0:
            raise ValueError("soft HF rollout requires branch_rollout_thinking_tokens or early_stopping_length_threshold")
        return thinking_tokens

    def _continuation_tokens(self, thinking_tokens: int) -> int:
        configured = int(self.config.get("branch_rollout_continuation_tokens", 0))
        if configured > 0:
            continuation_tokens = configured
        else:
            continuation_tokens = int(self.config.response_length) - thinking_tokens - 1
        if continuation_tokens < 0:
            raise ValueError(
                "soft HF rollout token budget exceeds response length: "
                f"thinking={thinking_tokens}, continuation={continuation_tokens}, response={self.config.response_length}"
            )
        if thinking_tokens + 1 + continuation_tokens > int(self.config.response_length):
            raise ValueError(
                "soft HF rollout token budget exceeds response length: "
                f"thinking={thinking_tokens}, continuation={continuation_tokens}, response={self.config.response_length}"
            )
        return continuation_tokens

    def _check_elapsed(self, step: int, total: int, phase: str) -> None:
        if self._debug and (step == 0 or (step + 1) % 128 == 0 or step + 1 == total):
            print(f"[soft_hf_rollout] {phase} step={step + 1}/{total}", flush=True)

    def _resolve_pad_token_id(self) -> int:
        pad = getattr(self.model_config.generation_config, "pad_token_id", None)
        if pad is None:
            pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if isinstance(pad, list):
            pad = pad[0]
        return int(pad)

    def _resolve_eos_token_id(self) -> int | list[int]:
        eos = getattr(self.model_config.generation_config, "eos_token_id", None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        return eos

    def _resolve_think_end_id(self) -> int:
        token_ids = self.tokenizer.encode(str(self.config.get("think_end_str", "</think>")), add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"could not tokenize think_end_str={self.config.get('think_end_str', '</think>')!r}")
        return int(token_ids[-1])
