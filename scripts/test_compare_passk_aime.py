#!/usr/bin/env python3
import argparse
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from compare_passk_aime import (
    ASSISTANT_THINK_PREFILL,
    Example,
    FIXED_PREFIX_NEVER_SWITCH_THINK_END,
    MultiplexSelfCheckResult,
    PromptBuildInfo,
    SampleRecord,
    base_sampling_params,
    build_local_requests_session,
    build_manifest_payload,
    build_pass_at_k_values,
    build_prefilled_prompt_ids,
    build_run_timing,
    child_sampling_params,
    compute_summary_tables,
    count_visible_gpus_from_env,
    detect_nvcc_path,
    ensure_localhost_no_proxy_env,
    fixed_prefix_sampling_params,
    format_wall_clock_seconds,
    load_examples,
    make_sample_record,
    make_server_args,
    make_target_dp_rank,
    parse_reasoning_prefix_token_values,
    resolve_runtime_config,
    RESOURCE_PROFILE_AUTO,
    RESOURCE_PROFILE_LEGACY_8GPU,
    RESOURCE_PROFILE_TWO_GPU_SAFE,
    run_baseline_for_prompt,
    score_response,
    StructuredEventLogger,
    verify_multiplex_runtime,
    write_summary_md,
)


class ComparePassKAimeTests(unittest.TestCase):
    def make_args(self, **overrides):
        values = {
            "model": "Qwen/Qwen3-4B",
            "benchmark": "aime2024",
            "max_k": 16,
            "methods": "baseline,shared_trace,standard_generation",
            "seed": 1234,
            "output_dir": "/tmp/out",
            "resume": False,
            "host": "127.0.0.1",
            "port": 30000,
            "api_key": "multiplex-local",
            "dp_size": 2,
            "tp_size": 1,
            "request_batch_size": None,
            "resource_profile": RESOURCE_PROFILE_AUTO,
            "capacity_of_str_len": 32768,
            "max_new_tokens": 8192,
            "server_timeout_seconds": 3600,
            "max_prompts": 50,
            "reasoning_prefix_token_values": "512,1024,2048",
            "checkpoint_matched_prompts_step": 5,
            "mem_fraction_static": None,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def make_example(self):
        return Example(
            prompt_index=0,
            problem_id="2024-I-1",
            problem="Problem",
            answer="42",
            prompt_ids=[11, 22, 33],
            assistant_prefill=ASSISTANT_THINK_PREFILL,
            prompt_build_mode="native_generation_prompt",
        )

    def test_build_pass_at_k_values_appends_non_power_of_two(self):
        self.assertEqual(build_pass_at_k_values(20), [1, 2, 4, 8, 16, 20])

    def test_parse_reasoning_prefix_token_values_deduplicates_and_preserves_order(self):
        self.assertEqual(
            parse_reasoning_prefix_token_values("512, 1024,512,2048"),
            [512, 1024, 2048],
        )

    def test_count_visible_gpus_from_env(self):
        self.assertEqual(count_visible_gpus_from_env("0,1"), 2)
        self.assertEqual(count_visible_gpus_from_env("GPU-aaa,GPU-bbb,GPU-ccc"), 3)
        self.assertEqual(count_visible_gpus_from_env(""), 0)
        self.assertIsNone(count_visible_gpus_from_env(None))

    def test_resolve_runtime_config_clamps_dp_size_to_visible_gpus(self):
        args = self.make_args(dp_size=8)
        runtime = resolve_runtime_config(
            args,
            visible_gpu_count=2,
            per_gpu_memory_gb=48.0,
            nvcc_path="/tmp/fake-nvcc",
        )
        self.assertEqual(runtime.requested_dp_size, 8)
        self.assertEqual(runtime.effective_dp_size, 2)
        self.assertEqual(runtime.resource_profile, RESOURCE_PROFILE_TWO_GPU_SAFE)
        self.assertEqual(runtime.request_batch_size, 16)
        self.assertTrue(any("dp_size=8" in message for message in runtime.clamp_messages))

    def test_resolve_runtime_config_uses_low_memory_safe_batch_cap(self):
        args = self.make_args(request_batch_size=32)
        runtime = resolve_runtime_config(
            args,
            visible_gpu_count=2,
            per_gpu_memory_gb=24.0,
            nvcc_path="/tmp/fake-nvcc",
        )
        self.assertEqual(runtime.resource_profile, RESOURCE_PROFILE_TWO_GPU_SAFE)
        self.assertEqual(runtime.safe_request_batch_cap, 8)
        self.assertEqual(runtime.request_batch_size, 8)

    def test_resolve_runtime_config_legacy_profile_keeps_defaults(self):
        args = self.make_args(
            dp_size=8,
            resource_profile=RESOURCE_PROFILE_LEGACY_8GPU,
        )
        runtime = resolve_runtime_config(
            args,
            visible_gpu_count=8,
            per_gpu_memory_gb=48.0,
            nvcc_path="/tmp/fake-nvcc",
        )
        self.assertEqual(runtime.resource_profile, RESOURCE_PROFILE_LEGACY_8GPU)
        self.assertEqual(runtime.request_batch_size, 64)
        self.assertIsNone(runtime.chunked_prefill_size)

    def test_resolve_runtime_config_uses_torch_native_without_nvcc_on_low_memory_gpus(self):
        args = self.make_args()
        runtime = resolve_runtime_config(
            args,
            visible_gpu_count=2,
            per_gpu_memory_gb=23.46,
            nvcc_path=None,
        )
        self.assertIsNone(runtime.nvcc_path)
        self.assertEqual(runtime.attention_backend, "torch_native")
        self.assertEqual(runtime.decode_attention_backend, "torch_native")
        self.assertEqual(runtime.prefill_attention_backend, "torch_native")
        self.assertEqual(runtime.sampling_backend, "pytorch")
        self.assertTrue(runtime.disable_cuda_graph)
        self.assertTrue(runtime.disable_radix_cache)
        self.assertEqual(runtime.chunked_prefill_size, -1)
        self.assertEqual(runtime.max_running_requests, runtime.request_batch_size)
        self.assertTrue(
            any("falling back to torch_native attention" in message for message in runtime.clamp_messages)
        )

    def test_resolve_runtime_config_prefers_triton_without_nvcc_on_higher_memory_gpus(self):
        args = self.make_args()
        runtime = resolve_runtime_config(
            args,
            visible_gpu_count=2,
            per_gpu_memory_gb=48.0,
            nvcc_path=None,
        )
        self.assertIsNone(runtime.nvcc_path)
        self.assertEqual(runtime.attention_backend, "triton")
        self.assertEqual(runtime.decode_attention_backend, "triton")
        self.assertEqual(runtime.prefill_attention_backend, "triton")
        self.assertIsNone(runtime.sampling_backend)
        self.assertTrue(runtime.disable_cuda_graph)
        self.assertFalse(runtime.disable_radix_cache)
        self.assertTrue(any("forcing Triton attention" in message for message in runtime.clamp_messages))

    def test_detect_nvcc_path_prefers_cudacxx_then_cuda_home_then_path(self):
        with mock.patch.dict(
            "os.environ",
            {
                "CUDACXX": "/tmp/toolchain/bin/nvcc",
                "CUDA_HOME": "/tmp/cuda-home",
            },
            clear=False,
        ), mock.patch(
            "compare_passk_aime.shutil.which",
            return_value="/tmp/from-path/nvcc",
        ), mock.patch(
            "compare_passk_aime.Path.is_file",
            return_value=True,
        ), mock.patch(
            "compare_passk_aime.os.access",
            return_value=True,
        ):
            self.assertEqual(detect_nvcc_path(), "/tmp/toolchain/bin/nvcc")

    def test_ensure_localhost_no_proxy_env_adds_loopback_entries(self):
        with mock.patch.dict(
            "os.environ",
            {"NO_PROXY": "example.com", "no_proxy": "internal.local"},
            clear=False,
        ):
            ensure_localhost_no_proxy_env()
            for env_name in ("NO_PROXY", "no_proxy"):
                value = os.environ[env_name]
                self.assertIn("127.0.0.1", value)
                self.assertIn("localhost", value)
                self.assertIn("::1", value)

    def test_build_local_requests_session_disables_proxy_env_lookup(self):
        class FakeSession:
            def __init__(self):
                self.trust_env = True

        fake_requests = types.SimpleNamespace(Session=FakeSession)
        session = build_local_requests_session(fake_requests)
        self.assertFalse(session.trust_env)

    def test_build_prefilled_prompt_ids_uses_native_generation_prompt(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt=False, continue_final_message=False):
                self.last_messages = messages
                self.last_continue = continue_final_message
                self.last_add_generation_prompt = add_generation_prompt
                if tokenize:
                    return [1, 2, 3]
                return "unused"

        tokenizer = FakeTokenizer()
        prompt_ids, prompt_info = build_prefilled_prompt_ids(tokenizer, "Problem text")
        self.assertEqual(prompt_ids, [1, 2, 3])
        self.assertEqual(prompt_info.prompt_build_mode, "native_generation_prompt")
        self.assertEqual(prompt_info.assistant_prefill, ASSISTANT_THINK_PREFILL)
        self.assertTrue(tokenizer.last_add_generation_prompt)
        self.assertFalse(tokenizer.last_continue)
        self.assertEqual(tokenizer.last_messages, [{"role": "user", "content": "Problem text"}])

    def test_build_prefilled_prompt_ids_falls_back_to_text_generation_prompt(self):
        class FakeTokenizer:
            def __init__(self):
                self.encoded_text = None

            def apply_chat_template(self, messages, tokenize, add_generation_prompt=False, continue_final_message=False):
                if tokenize:
                    raise ValueError("unsupported")
                return "<user>Problem</user><assistant><think>\n"

            def encode(self, text, add_special_tokens=False):
                self.encoded_text = text
                return [7, 8, 9]

        tokenizer = FakeTokenizer()
        prompt_ids, prompt_info = build_prefilled_prompt_ids(tokenizer, "Problem")
        self.assertEqual(prompt_ids, [7, 8, 9])
        self.assertEqual(prompt_info.prompt_build_mode, "native_generation_prompt_text")
        self.assertEqual(tokenizer.encoded_text, "<user>Problem</user><assistant><think>\n")

    def test_load_examples_accepts_string_problem_ids_and_returns_prompt_build_info(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt=False, continue_final_message=False):
                return [11, 22, 33]

        fake_tokenizer = FakeTokenizer()
        fake_datasets = types.SimpleNamespace(
            load_dataset=lambda name, split: [
                {"ID": "2024-II-4", "Problem": "What is 2+2?", "Answer": "4"},
            ]
        )

        with mock.patch.dict(sys.modules, {"datasets": fake_datasets}):
            examples, prompt_build_info = load_examples(fake_tokenizer, "aime2024", None)

        self.assertEqual(len(examples), 1)
        self.assertEqual(examples[0].problem_id, "2024-II-4")
        self.assertEqual(examples[0].assistant_prefill, ASSISTANT_THINK_PREFILL)
        self.assertEqual(prompt_build_info.prompt_build_mode, "native_generation_prompt")

    def test_score_response_uses_exact_boxed_integer_match(self):
        fake_math_verify = types.SimpleNamespace(
            parse=lambda text, parsing_timeout: ["parsed", "4"],
            verify=lambda left, right, timeout_seconds: False,
        )

        with mock.patch.dict(sys.modules, {"math_verify": fake_math_verify}):
            score_result = score_response("<think>reason</think> \\boxed{4}", "4")

        self.assertEqual(score_result.score, 1.0)
        self.assertEqual(score_result.reason, "exact_integer_match")
        self.assertEqual(score_result.extracted_answer, "4")

    def test_score_response_uses_symbolic_verify_fallback(self):
        def fake_parse(text, parsing_timeout):
            if text.startswith("\\boxed{"):
                return ["expected", "4"]
            return ["parsed", "not-a-literal-match"]

        fake_math_verify = types.SimpleNamespace(
            parse=fake_parse,
            verify=lambda left, right, timeout_seconds: True,
        )

        with mock.patch.dict(sys.modules, {"math_verify": fake_math_verify}):
            score_result = score_response("<think>reason</think> final answer text", "4")

        self.assertEqual(score_result.score, 1.0)
        self.assertEqual(score_result.reason, "symbolic_verify_match")

    def test_score_response_falls_back_when_think_end_is_missing(self):
        fake_math_verify = types.SimpleNamespace(
            parse=lambda text, parsing_timeout: ["parsed", "17"],
            verify=lambda left, right, timeout_seconds: False,
        )
        with mock.patch.dict(sys.modules, {"math_verify": fake_math_verify}):
            score_result = score_response("Final answer: \\boxed{17}", "17")
        self.assertEqual(score_result.score, 1.0)
        self.assertEqual(score_result.extracted_answer, "17")

    def test_make_sample_record_marks_max_length_samples_unusable(self):
        fake_math_verify = types.SimpleNamespace(
            parse=lambda text, parsing_timeout: ["parsed", "42"],
            verify=lambda left, right, timeout_seconds: False,
        )
        with mock.patch.dict(sys.modules, {"math_verify": fake_math_verify}):
            record = make_sample_record(
                benchmark="aime2024",
                method="baseline_independent",
                example=self.make_example(),
                sample_index=0,
                rid="rid-0",
                target_dp_rank=0,
                output={
                    "text": "\\boxed{42}",
                    "meta_info": {
                        "finish_reason": "length",
                        "prompt_tokens": 3,
                        "completion_tokens": 4096,
                        "cached_tokens": 0,
                        "e2e_latency": 0.1,
                    },
                },
                max_new_tokens=4096,
            )
        self.assertTrue(record.usable_for_eval)
        self.assertIsNone(record.excluded_reason)
        self.assertEqual(record.generated_suffix, "\\boxed{42}")
        self.assertEqual(record.score_reason, "exact_integer_match")

    def test_make_sample_record_scores_missing_think_end_output(self):
        fake_math_verify = types.SimpleNamespace(
            parse=lambda text, parsing_timeout: ["parsed", "42"],
            verify=lambda left, right, timeout_seconds: False,
        )
        with mock.patch.dict(sys.modules, {"math_verify": fake_math_verify}):
            record = make_sample_record(
                benchmark="aime2024",
                method="baseline_independent",
                example=self.make_example(),
                sample_index=0,
                rid="rid-0",
                target_dp_rank=0,
                output={
                    "text": "reasoning without a close tag",
                    "meta_info": {
                        "finish_reason": {"type": "stop", "matched": 151645},
                        "prompt_tokens": 3,
                        "completion_tokens": 128,
                        "cached_tokens": 0,
                        "e2e_latency": 0.1,
                    },
                },
                max_new_tokens=4096,
            )
        self.assertTrue(record.usable_for_eval)
        self.assertIsNone(record.excluded_reason)
        self.assertEqual(record.score_reason, "parsed_direct_match")

    def test_make_server_args_disables_unused_grammar_backend(self):
        args = self.make_args()
        runtime = resolve_runtime_config(
            args,
            visible_gpu_count=2,
            per_gpu_memory_gb=48.0,
            nvcc_path=None,
        )

        class FakeServerArgs:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        fake_sglang = types.ModuleType("sglang")
        fake_srt = types.ModuleType("sglang.srt")
        fake_server_args_module = types.ModuleType("sglang.srt.server_args")
        fake_sglang.srt = fake_srt
        fake_srt.server_args = fake_server_args_module
        fake_server_args_module.ServerArgs = FakeServerArgs

        with mock.patch.dict(
            sys.modules,
            {
                "sglang": fake_sglang,
                "sglang.srt": fake_srt,
                "sglang.srt.server_args": fake_server_args_module,
            },
        ):
            server_args = make_server_args(args, runtime)

        self.assertEqual(server_args.grammar_backend, "none")
        self.assertEqual(server_args.enable_soft_thinking, True)
        self.assertEqual(server_args.think_end_str, "</think>")
        self.assertEqual(server_args.reasoning_parser, "qwen3")
        self.assertEqual(server_args.attention_backend, "triton")
        self.assertTrue(server_args.disable_overlap_schedule)
        self.assertTrue(server_args.disable_cuda_graph)

    def test_make_server_args_uses_torch_native_profile_on_low_memory_no_nvcc_nodes(self):
        args = self.make_args()
        runtime = resolve_runtime_config(
            args,
            visible_gpu_count=2,
            per_gpu_memory_gb=23.46,
            nvcc_path=None,
        )

        class FakeServerArgs:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        fake_sglang = types.ModuleType("sglang")
        fake_srt = types.ModuleType("sglang.srt")
        fake_server_args_module = types.ModuleType("sglang.srt.server_args")
        fake_sglang.srt = fake_srt
        fake_srt.server_args = fake_server_args_module
        fake_server_args_module.ServerArgs = FakeServerArgs

        with mock.patch.dict(
            sys.modules,
            {
                "sglang": fake_sglang,
                "sglang.srt": fake_srt,
                "sglang.srt.server_args": fake_server_args_module,
            },
        ):
            server_args = make_server_args(args, runtime)

        self.assertEqual(server_args.attention_backend, "torch_native")
        self.assertEqual(server_args.decode_attention_backend, "torch_native")
        self.assertEqual(server_args.prefill_attention_backend, "torch_native")
        self.assertEqual(server_args.sampling_backend, "pytorch")
        self.assertEqual(server_args.chunked_prefill_size, -1)
        self.assertEqual(server_args.max_running_requests, runtime.request_batch_size)
        self.assertTrue(server_args.disable_radix_cache)

    def test_base_sampling_params_use_tighter_thinking_distribution(self):
        params = base_sampling_params(128)
        self.assertEqual(params["temperature"], 0.8)
        self.assertEqual(params["top_p"], 0.95)
        self.assertEqual(params["after_thinking_temperature"], 0.6)
        self.assertEqual(params["after_thinking_top_p"], 0.95)

    def test_child_sampling_params_match_baseline_after_thinking_settings(self):
        baseline_params = base_sampling_params(128)
        child_params = child_sampling_params(128)
        self.assertEqual(child_params["temperature"], baseline_params["after_thinking_temperature"])
        self.assertEqual(child_params["top_p"], baseline_params["after_thinking_top_p"])
        self.assertEqual(child_params["top_k"], baseline_params["after_thinking_top_k"])
        self.assertEqual(child_params["min_p"], baseline_params["after_thinking_min_p"])

    def test_fixed_prefix_sampling_params_prevents_early_think_end_switch(self):
        params = fixed_prefix_sampling_params(512)
        self.assertEqual(params["max_new_tokens"], 512)
        self.assertEqual(params["think_end_str"], FIXED_PREFIX_NEVER_SWITCH_THINK_END)

    def test_build_manifest_payload_records_prompt_and_self_check_metadata(self):
        args = self.make_args(dp_size=8, request_batch_size=32)
        runtime = resolve_runtime_config(
            args,
            visible_gpu_count=2,
            per_gpu_memory_gb=24.0,
            nvcc_path=None,
        )
        with mock.patch(
            "compare_passk_aime.make_server_args",
            return_value=argparse.Namespace(dp_size=2, tp_size=1, request_batch_size=8),
        ):
            manifest = build_manifest_payload(
                args=args,
                runtime_config=runtime,
                pass_at_ks=[1, 2],
                examples=[self.make_example()],
                methods=["baseline_independent"],
                prompt_build_info=PromptBuildInfo(
                    assistant_prefill=ASSISTANT_THINK_PREFILL,
                    prompt_build_mode="native_generation_prompt",
                ),
                multiplex_self_check=MultiplexSelfCheckResult(
                    success=True,
                    message="ok",
                    attention_backend="triton",
                    enable_soft_thinking=True,
                    has_topk_metadata=True,
                    finish_reason={"type": "stop", "matched": "</think>"},
                    output_text="<think>\nreason</think>",
                ),
                summary_json={
                    "eligible_prompts_by_k": {"1": [0]},
                    "exclusion_counts": {"max_new_tokens_reached": 2},
                },
                run_timing=build_run_timing(100.0, 112.5),
            )
        self.assertEqual(manifest["requested_dp_size"], 8)
        self.assertEqual(manifest["dp_size"], 2)
        self.assertEqual(manifest["assistant_prefill"], ASSISTANT_THINK_PREFILL)
        self.assertEqual(manifest["prompt_build_mode"], "native_generation_prompt")
        self.assertEqual(manifest["run_status"], "running")
        self.assertEqual(manifest["reasoning_prefix_token_values"], [512, 1024, 2048])
        self.assertTrue(manifest["multiplex_self_check"]["success"])
        self.assertEqual(manifest["summary_metadata"]["exclusion_counts"]["max_new_tokens_reached"], 2)
        self.assertEqual(manifest["run_timing"]["started_at_unix"], 100.0)
        self.assertEqual(manifest["run_timing"]["finished_at_unix"], 112.5)
        self.assertEqual(manifest["run_timing"]["wall_clock_hms"], "0:00:12")
        self.assertEqual(manifest["sampling_defaults"]["temperature"], 0.8)
        self.assertEqual(manifest["sampling_defaults"]["top_p"], 0.95)
        self.assertIn("512", manifest["fixed_prefix_sampling_defaults"])

    def test_runner_relies_on_native_generation_prompt_without_manual_think_prefill(self):
        source = (SCRIPT_DIR / "compare_passk_aime.py").read_text(encoding="utf-8")
        self.assertIn('ASSISTANT_THINK_PREFILL = ""', source)
        self.assertIn('prompt_build_mode="native_generation_prompt"', source)

    def test_write_summary_md_includes_wall_clock_section(self):
        summary_rows = [
            {
                "method": "baseline_independent",
                "k": 1,
                "num_prompts": 2,
                "pass_at_k": 0.5,
                "avg_cost_tokens": 123.0,
                "num_failures": 0,
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "summary.md"
            write_summary_md(
                path,
                summary_rows,
                summary_rows,
                [1],
                summary_json={
                    "run_timing": {
                        "started_at": "2026-04-19T00:00:00+00:00",
                        "finished_at": "2026-04-19T01:02:03+00:00",
                        "wall_clock_seconds": 3723.0,
                        "wall_clock_hms": format_wall_clock_seconds(3723.0),
                    }
                },
            )
            text = path.read_text(encoding="utf-8")
        self.assertIn("## Wall Clock", text)
        self.assertIn("- duration: 1:02:03 (3723.0 seconds)", text)
        self.assertIn("- started_at: 2026-04-19T00:00:00+00:00", text)
        self.assertIn("- finished_at: 2026-04-19T01:02:03+00:00", text)

    def test_run_baseline_for_prompt_generates_fixed_prefix_then_continuation(self):
        args = self.make_args(
            max_k=2,
            max_new_tokens=32,
        )
        args.effective_request_batch_size = 2
        args.current_reasoning_prefix_tokens = 8
        example = self.make_example()

        class FakeClient:
            def __init__(self):
                self.generate_calls = []
                self.fork_calls = []
                self.opened_sessions = []
                self.closed_sessions = []
                self.outputs = [
                    [
                        {
                            "text": "prefix-a",
                            "meta_info": {
                                "finish_reason": {"type": "length"},
                                "prompt_tokens": 3,
                                "completion_tokens": 8,
                                "cached_tokens": 0,
                                "e2e_latency": 0.1,
                            },
                        },
                        {
                            "text": "prefix-b",
                            "meta_info": {
                                "finish_reason": {"type": "length"},
                                "prompt_tokens": 3,
                                "completion_tokens": 8,
                                "cached_tokens": 0,
                                "e2e_latency": 0.1,
                            },
                        },
                    ],
                    [
                        {
                            "text": " \\boxed{42}",
                            "meta_info": {
                                "finish_reason": {"type": "stop", "matched": 151645},
                                "prompt_tokens": 11,
                                "completion_tokens": 4,
                                "cached_tokens": 8,
                                "e2e_latency": 0.2,
                            },
                        },
                        {
                            "text": " \\boxed{42}",
                            "meta_info": {
                                "finish_reason": {"type": "stop", "matched": 151645},
                                "prompt_tokens": 11,
                                "completion_tokens": 5,
                                "cached_tokens": 8,
                                "e2e_latency": 0.2,
                            },
                        },
                    ],
                ]

            def open_session(self, capacity_of_str_len, session_id=None):
                self.opened_sessions.append((capacity_of_str_len, session_id))
                return session_id

            def close_session(self, session_id):
                self.closed_sessions.append(session_id)

            def fork_request(
                self,
                session_id,
                parent_rid,
                child_count,
                child_rids,
                child_seeds,
                target_dp_rank,
                allow_non_eot_branch=False,
            ):
                self.fork_calls.append(
                    {
                        "session_id": session_id,
                        "parent_rid": parent_rid,
                        "child_count": child_count,
                        "child_rids": child_rids,
                        "child_seeds": child_seeds,
                        "target_dp_rank": target_dp_rank,
                        "allow_non_eot_branch": allow_non_eot_branch,
                    }
                )
                return {
                    "success": True,
                    "message": "ok",
                    "branch_input_ids": [101, 102],
                    "cacheable_input_ids": [1, 2],
                    "uncached_tail_input_ids": [102],
                    "cacheable_token_count": 1,
                }

            def generate(self, **kwargs):
                self.generate_calls.append(kwargs)
                return self.outputs[len(self.generate_calls) - 1]

        fake_math_verify = types.SimpleNamespace(
            parse=lambda text, parsing_timeout: ["parsed", "42"],
            verify=lambda left, right, timeout_seconds: False,
        )

        client = FakeClient()
        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.dict(
            sys.modules,
            {"math_verify": fake_math_verify},
        ):
            event_path = Path(tmpdir) / "events.jsonl"
            baseline_result, records, attempt_records = run_baseline_for_prompt(
                args=args,
                benchmark="aime2024",
                client=client,
                example=example,
                target_dp_rank=0,
                logger=mock.Mock(),
                event_logger=StructuredEventLogger(event_path),
            )
            event_lines = event_path.read_text(encoding="utf-8").splitlines()

        self.assertTrue(baseline_result.success)
        self.assertEqual(len(records), 2)
        self.assertEqual(len(attempt_records), 0)
        self.assertEqual(len(client.generate_calls), 2)
        self.assertEqual(len(client.fork_calls), 2)
        self.assertTrue(all(call["allow_non_eot_branch"] for call in client.fork_calls))
        self.assertIsInstance(client.generate_calls[0]["session_params"], list)
        self.assertEqual(records[0].prefix_completion_tokens, 8)
        self.assertEqual(records[0].reasoning_prefix_tokens, 8)
        self.assertEqual(records[0].text, "prefix-a \\boxed{42}")
        self.assertTrue(any('"tag": "baseline_prefix_sample"' in line for line in event_lines))
        self.assertTrue(any('"tag": "baseline_continuation_sample"' in line for line in event_lines))

    def test_verify_multiplex_runtime_fails_when_soft_thinking_is_disabled(self):
        class FakeTokenizer:
            def apply_chat_template(self, messages, tokenize, add_generation_prompt=False, continue_final_message=False):
                return [1, 2, 3]

        class FakeClient:
            def get_server_info(self):
                return {"enable_soft_thinking": False, "attention_backend": "flashinfer"}

        with tempfile.TemporaryDirectory() as tmpdir:
            event_logger = StructuredEventLogger(Path(tmpdir) / "events.jsonl")
            result = verify_multiplex_runtime(
                client=FakeClient(),
                tokenizer=FakeTokenizer(),
                args=self.make_args(),
                logger=mock.Mock(),
                event_logger=event_logger,
            )
        self.assertFalse(result.success)
        self.assertIn("enable_soft_thinking=False", result.message)

    def test_compute_summary_tables_uses_only_matched_usable_samples(self):
        sample_rows = [
            {
                "method": "baseline_independent",
                "prompt_index": 0,
                "sample_index": 0,
                "correct": True,
                "completion_tokens": 3,
                "usable_for_eval": True,
            },
            {
                "method": "baseline_independent",
                "prompt_index": 0,
                "sample_index": 1,
                "correct": False,
                "completion_tokens": 4,
                "usable_for_eval": True,
            },
            {
                "method": "shared_trace_branch_after_prefix",
                "prompt_index": 0,
                "sample_index": 0,
                "correct": False,
                "completion_tokens": 5,
                "usable_for_eval": True,
            },
            {
                "method": "shared_trace_branch_after_prefix",
                "prompt_index": 0,
                "sample_index": 1,
                "correct": True,
                "completion_tokens": 6,
                "usable_for_eval": True,
            },
            {
                "method": "baseline_independent",
                "prompt_index": 1,
                "sample_index": 0,
                "correct": True,
                "completion_tokens": 3,
                "usable_for_eval": False,
                "excluded_reason": "max_new_tokens_reached",
                "finish_reason": "length",
            },
            {
                "method": "shared_trace_branch_after_prefix",
                "prompt_index": 1,
                "sample_index": 0,
                "correct": True,
                "completion_tokens": 5,
                "usable_for_eval": True,
            },
        ]
        parent_rows = [
            {
                "prompt_index": 0,
                "success": True,
                "completion_tokens": 7,
                "usable_for_eval": True,
            },
            {
                "prompt_index": 1,
                "success": False,
                "completion_tokens": 2,
                "usable_for_eval": False,
                "excluded_reason": "missing_think_end",
                "message": "missing eot",
            },
        ]

        summary_rows, per_prompt_rows, summary_json = compute_summary_tables(
            sample_rows=sample_rows,
            baseline_prompt_rows=[
                {
                    "prompt_index": 0,
                    "success": True,
                    "attempts_used": 2,
                    "total_completion_tokens_spent": 7,
                    "total_latency_seconds_spent": 0.2,
                    "reject_reason_counts": {},
                    "message": "ok",
                },
                {
                    "prompt_index": 1,
                    "success": False,
                    "attempts_used": 2,
                    "total_completion_tokens_spent": 3,
                    "total_latency_seconds_spent": 0.1,
                    "reject_reason_counts": {"max_new_tokens_reached": 1},
                    "message": "budget exhausted",
                },
            ],
            parent_rows=parent_rows,
            attempt_rows=[],
            pass_at_ks=[1, 2],
            max_k=8,
        )

        self.assertEqual({row["prompt_index"] for row in per_prompt_rows}, {0})
        baseline_k1 = next(
            row for row in summary_rows if row["method"] == "baseline_independent" and row["k"] == 1
        )
        baseline_k2 = next(
            row for row in summary_rows if row["method"] == "baseline_independent" and row["k"] == 2
        )
        self.assertEqual(baseline_k1["num_prompts"], 1)
        self.assertAlmostEqual(baseline_k1["pass_at_k"], 1.0)
        self.assertAlmostEqual(baseline_k2["pass_at_k"], 1.0)
        self.assertEqual(summary_json["eligible_prompts_by_k"]["1"], [0])
        self.assertEqual(summary_json["eligible_prompts_by_k"]["2"], [0])
        self.assertEqual(summary_json["exclusion_counts"]["baseline_independent_no_usable_samples"], 1)
        self.assertEqual(summary_json["exclusion_counts"]["missing_think_end"], 1)
        self.assertEqual(summary_json["coverage"]["matched_prompts_with_full_k"], 0)

    def test_compute_summary_tables_caps_shared_prompt_at_minimum_usable_k(self):
        sample_rows = []
        for sample_index in range(8):
            sample_rows.append(
                {
                    "method": "baseline_independent",
                    "prompt_index": 0,
                    "sample_index": sample_index,
                    "correct": sample_index == 0,
                    "completion_tokens": 1,
                    "usable_for_eval": True,
                }
            )
        for sample_index in range(5):
            sample_rows.append(
                {
                    "method": "shared_trace_branch_after_prefix",
                    "prompt_index": 0,
                    "sample_index": sample_index,
                    "correct": sample_index == 0,
                    "completion_tokens": 1,
                    "usable_for_eval": True,
                }
            )
        parent_rows = [
            {
                "prompt_index": 0,
                "success": True,
                "completion_tokens": 2,
                "usable_for_eval": True,
            }
        ]
        summary_rows, per_prompt_rows, summary_json = compute_summary_tables(
            sample_rows=sample_rows,
            baseline_prompt_rows=[
                {
                    "prompt_index": 0,
                    "success": True,
                    "attempts_used": 8,
                    "total_completion_tokens_spent": 8,
                    "total_latency_seconds_spent": 0.8,
                    "reject_reason_counts": {},
                    "message": "ok",
                }
            ],
            parent_rows=parent_rows,
            attempt_rows=[],
            pass_at_ks=[1, 4, 8],
            max_k=8,
        )
        self.assertEqual(per_prompt_rows[0]["matched_usable_samples"], 5)
        self.assertEqual(summary_json["eligible_prompts_by_k"]["1"], [0])
        self.assertEqual(summary_json["eligible_prompts_by_k"]["4"], [0])
        self.assertEqual(summary_json["eligible_prompts_by_k"]["8"], [])
        baseline_k8 = next(
            row for row in summary_rows if row["method"] == "baseline_independent" and row["k"] == 8
        )
        self.assertEqual(baseline_k8["num_prompts"], 0)
        self.assertIsNone(baseline_k8["pass_at_k"])
        self.assertEqual(summary_json["coverage"]["baseline_prompts_with_full_k"], 1)
        self.assertEqual(summary_json["coverage"]["shared_trace_prompts_with_full_k"], 0)


if __name__ == "__main__":
    unittest.main()
