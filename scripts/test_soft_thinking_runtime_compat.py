#!/usr/bin/env python3
import ast
import logging
from pathlib import Path
import typing
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
OVERLAP_THREAD_PATH = (
    REPO_ROOT
    / "sglang-0.4.9.post6"
    / "sglang"
    / "srt"
    / "managers"
    / "tp_worker_overlap_thread.py"
)
TP_WORKER_PATH = (
    REPO_ROOT
    / "sglang-0.4.9.post6"
    / "sglang"
    / "srt"
    / "managers"
    / "tp_worker.py"
)
HTTP_SERVER_PATH = (
    REPO_ROOT
    / "sglang-0.4.9.post6"
    / "sglang"
    / "srt"
    / "entrypoints"
    / "http_server.py"
)
MODEL_RUNNER_PATH = (
    REPO_ROOT
    / "sglang-0.4.9.post6"
    / "sglang"
    / "srt"
    / "model_executor"
    / "model_runner.py"
)
CUDA_GRAPH_RUNNER_PATH = (
    REPO_ROOT
    / "sglang-0.4.9.post6"
    / "sglang"
    / "srt"
    / "model_executor"
    / "cuda_graph_runner.py"
)
SCHEDULER_OUTPUT_PROCESSOR_PATH = (
    REPO_ROOT
    / "sglang-0.4.9.post6"
    / "sglang"
    / "srt"
    / "managers"
    / "scheduler_output_processor_mixin.py"
)
SCHEDULER_PATH = (
    REPO_ROOT
    / "sglang-0.4.9.post6"
    / "sglang"
    / "srt"
    / "managers"
    / "scheduler.py"
)
SAMPLER_PATH = (
    REPO_ROOT
    / "sglang-0.4.9.post6"
    / "sglang"
    / "srt"
    / "layers"
    / "sampler.py"
)
VOCAB_PARALLEL_EMBEDDING_PATH = (
    REPO_ROOT
    / "sglang-0.4.9.post6"
    / "sglang"
    / "srt"
    / "layers"
    / "vocab_parallel_embedding.py"
)
TOKENIZER_MANAGER_PATH = (
    REPO_ROOT
    / "sglang-0.4.9.post6"
    / "sglang"
    / "srt"
    / "managers"
    / "tokenizer_manager.py"
)


def load_function_from_source(path: Path, function_name: str):
    module_ast = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in module_ast.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            isolated_module = ast.Module(body=[node], type_ignores=[])
            isolated_module = ast.fix_missing_locations(isolated_module)
            namespace = {
                "Optional": typing.Optional,
                "logger": logging.getLogger("test_soft_thinking_runtime_compat"),
            }
            exec(compile(isolated_module, str(path), "exec"), namespace)
            return namespace[function_name]
    raise AssertionError(f"Function {function_name!r} was not found in {path}")


class SoftThinkingRuntimeCompatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.resolve_think_end_token_id = staticmethod(
            load_function_from_source(
                OVERLAP_THREAD_PATH,
                "resolve_think_end_token_id",
            )
        )

    def test_resolve_think_end_token_id_uses_last_token(self):
        class FakeTokenizer:
            def encode(self, text, add_special_tokens=False):
                self.last = (text, add_special_tokens)
                return [11, 22, 33]

        tokenizer = FakeTokenizer()
        token_id = self.resolve_think_end_token_id("</think>", tokenizer)
        self.assertEqual(token_id, 33)
        self.assertEqual(tokenizer.last, ("</think>", False))

    def test_resolve_think_end_token_id_uses_fallback_tokenizer(self):
        class FallbackTokenizer:
            def encode(self, text, add_special_tokens=False):
                return [7]

        token_id = self.resolve_think_end_token_id(
            "</think>",
            primary_tokenizer=None,
            fallback_tokenizer=FallbackTokenizer(),
        )
        self.assertEqual(token_id, 7)

    def test_resolve_think_end_token_id_returns_minus_one_when_unavailable(self):
        token_id = self.resolve_think_end_token_id(
            "</think>",
            primary_tokenizer=None,
            fallback_tokenizer=None,
        )
        self.assertEqual(token_id, -1)

    def test_tp_worker_keeps_model_runner_tokenizer_compat_assignment(self):
        source = TP_WORKER_PATH.read_text(encoding="utf-8")
        self.assertIn("self.tokenizer = None", source)
        self.assertIn("self.processor = None", source)
        self.assertIn('self.model_runner.tokenizer = getattr(self, "tokenizer", None)', source)
        self.assertIn('self.model_runner.processor = getattr(self, "processor", None)', source)

    def test_server_warmup_disables_soft_thinking_and_avoids_greedy_mode(self):
        source = HTTP_SERVER_PATH.read_text(encoding="utf-8")
        self.assertIn("warmup_temperature = 0.6 if server_args.enable_soft_thinking else 0", source)
        self.assertIn('"custom_params": {"__disable_soft_thinking__": True}', source)

    def test_health_generate_disables_soft_thinking_and_avoids_greedy_mode(self):
        source = HTTP_SERVER_PATH.read_text(encoding="utf-8")
        self.assertIn("health_temperature = 0.6 if server_args.enable_soft_thinking else 0.0", source)
        self.assertIn('"custom_params": {"__disable_soft_thinking__": True}', source)

    def test_model_runner_decode_uses_batch_soft_thinking_state(self):
        source = MODEL_RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("batch_soft_thinking_active = (", source)
        self.assertIn("forward_batch.sampling_info.soft_thinking_modes", source)
        self.assertIn("torch.any(forward_batch.sampling_info.soft_thinking_modes).item()", source)
        self.assertIn("forward_batch.topk_probs is not None", source)
        self.assertIn("forward_batch.topk_indices is not None", source)
        self.assertIn("return self.model.forward(\n                forward_batch.input_ids", source)

    def test_model_runner_skips_cuda_graph_init_when_disabled(self):
        source = MODEL_RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("if self.server_args.disable_cuda_graph:\n            return", source)
        self.assertIn("self.cuda_graph_runner = CudaGraphRunner(self)", source)

    def test_scheduler_output_processing_uses_request_soft_thinking_state(self):
        source = SCHEDULER_OUTPUT_PROCESSOR_PATH.read_text(encoding="utf-8")
        self.assertIn("req.enable_soft_thinking", source)
        self.assertIn("logits_output.topk_probs is not None", source)
        self.assertIn("logits_output.topk_indices is not None", source)

    def test_scheduler_decode_bookkeeping_normalizes_per_request_soft_modes(self):
        source = SCHEDULER_PATH.read_text(encoding="utf-8")
        self.assertIn("batch.sampling_info.soft_thinking_modes is not None", source)
        self.assertIn("req.enable_soft_thinking", source)
        self.assertIn("torch.tensor(", source)

    def test_cuda_graph_runner_still_allocates_soft_thinking_buffers(self):
        source = CUDA_GRAPH_RUNNER_PATH.read_text(encoding="utf-8")
        self.assertIn("self.enable_soft_thinking = self.model_runner.server_args.enable_soft_thinking", source)
        self.assertIn("self.topk_probs = torch.zeros((self.max_bs, self.used_topk)", source)

    def test_sampler_activates_soft_thinking_only_when_any_request_uses_it(self):
        source = SAMPLER_PATH.read_text(encoding="utf-8")
        self.assertIn("soft_thinking_active = (", source)
        self.assertIn("sampling_info.enable_soft_thinking", source)
        self.assertIn("torch.any(sampling_info.soft_thinking_modes).item()", source)
        self.assertIn("if soft_thinking_active:", source)

    def test_sampler_slow_soft_thinking_path_samples_without_min_p(self):
        source = SAMPLER_PATH.read_text(encoding="utf-8")
        self.assertIn(
            "if sampling_info.need_min_p_sampling or sampling_info.need_after_thinking_min_p_sampling: # slow",
            source,
        )
        self.assertIn(
            "assert sampling_info.used_topk == sampling_info.max_topk",
            source,
        )
        self.assertIn(
            "batch_next_token_ids = topk_indices[:, 0].to(torch.int32)",
            source,
        )

    def test_sampler_pytorch_backend_materializes_soft_thinking_topk_state(self):
        source = SAMPLER_PATH.read_text(encoding="utf-8")
        self.assertIn('elif global_server_args_dict["sampling_backend"] == "pytorch":', source)
        self.assertIn("if soft_thinking_active:", source)
        self.assertIn("logits_output.topk_probs = topk_probs", source)
        self.assertIn("logits_output.topk_indices = topk_indices", source)
        self.assertIn(
            "sampled_token_ids = top_k_top_p_min_p_sampling_from_probs_torch(",
            source,
        )

    def test_tp_worker_transitions_think_end_to_one_hot_decode_state(self):
        source = OVERLAP_THREAD_PATH.read_text(encoding="utf-8")
        self.assertIn("except MultiplexFidelityViolation as exc:", source)
        self.assertIn("self.output_queue.put(exc)", source)
        self.assertIn("sampling_info.soft_thinking_modes[update_mask] = False", source)
        self.assertIn("topk_probs[update_mask] = 0.0", source)
        self.assertIn("topk_indices[update_mask] = 0", source)
        self.assertIn("topk_probs[update_mask, 0] = 1.0", source)
        self.assertIn("topk_indices[update_mask, 0] = input_ids[update_mask].long()", source)

    def test_weighted_embedding_sanitizes_malformed_topk_rows(self):
        source = VOCAB_PARALLEL_EMBEDDING_PATH.read_text(encoding="utf-8")
        self.assertIn("class MultiplexFidelityViolation(RuntimeError):", source)
        self.assertIn("def validate_weighted_topk_inputs(", source)
        self.assertIn("(topk_probs < 0).any(dim=-1)", source)
        self.assertIn("raise MultiplexFidelityViolation(", source)
        self.assertIn("topk_probs = validate_weighted_topk_inputs(", source)

    def test_tokenizer_manager_sizes_soft_thinking_chunks_without_logprobs(self):
        source = TOKENIZER_MANAGER_PATH.read_text(encoding="utf-8")
        self.assertIn("last_output_topk_completion_tokens: int = 0", source)
        self.assertIn("output_token_logprobs_val = self._batch_entry_or_empty(", source)
        self.assertIn("completion_tokens = self._get_recv_obj_completion_tokens(", source)
        self.assertIn(
            "cur_output_len = max(\n                            completion_tokens\n                            - state.last_output_topk_completion_tokens,",
            source,
        )
        self.assertIn("state.last_output_topk_completion_tokens = completion_tokens", source)

    def test_tokenizer_manager_normalizes_missing_logprob_payloads(self):
        source = TOKENIZER_MANAGER_PATH.read_text(encoding="utf-8")
        self.assertIn("def _batch_entry_or_empty(batch_values, recv_obj_index: int):", source)
        self.assertIn("if batch_values is None:", source)
        self.assertIn("return [] if value is None else value", source)
        self.assertIn(
            "output_token_logprobs_val = self._batch_entry_or_empty(\n            recv_obj.output_token_logprobs_val, recv_obj_index",
            source,
        )
        self.assertIn(
            "output_top_logprobs_val = self._batch_entry_or_empty(\n                recv_obj.output_top_logprobs_val, recv_obj_index",
            source,
        )
        self.assertIn(
            "output_token_ids_logprobs_val = self._batch_entry_or_empty(\n                recv_obj.output_token_ids_logprobs_val, recv_obj_index",
            source,
        )


if __name__ == "__main__":
    unittest.main()
