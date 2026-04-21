#!/usr/bin/env python3
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

SGLANG_ROOT = SCRIPT_DIR.parent / "sglang-0.4.9.post6"

from bootstrap_runtime_overlay import (
    DEFAULT_IMPORT_TARGETS,
    MANAGED_RUNTIME_SUPPORT_REQUIREMENTS,
    OVERLAY_CACHE_MAX_COPY_BYTES,
    ProbeResult,
    RuntimePackageSpec,
    build_isolated_pythonpath,
    build_lightweight_constraint_requirements,
    build_lightweight_overlay_key,
    bootstrap_runtime,
    build_managed_runtime_specs,
    build_requirement_index,
    build_runtime_key,
    candidate_requirements_for_module,
    changed_overlay_entry_names,
    collect_requirement_index,
    infer_requirements_from_traceback,
    install_requirement,
    overlay_entry_size_bytes,
    overlay_entry_requirement_key,
    probe_import_targets,
    purge_blocked_overlay_entries,
    runtime_executable_candidate_paths,
    runtime_manifest_path,
    runtime_site_packages_dir,
    select_lightweight_overlay_dir,
    select_runtime_root,
    should_cache_overlay_entry,
    snapshot_overlay_entries,
    sync_overlay_entry_names,
    sync_overlay_entries,
    validate_managed_runtime_environment,
    validate_manifest_payload,
)

runtime_compat_spec = importlib.util.spec_from_file_location(
    "runtime_compat_for_test",
    SGLANG_ROOT / "sglang" / "srt" / "runtime_compat.py",
)
runtime_compat_module = importlib.util.module_from_spec(runtime_compat_spec)
assert runtime_compat_spec.loader is not None
runtime_compat_spec.loader.exec_module(runtime_compat_module)
safe_cpu_has_amx_support = runtime_compat_module.cpu_has_amx_support

torchao_utils_spec = importlib.util.spec_from_file_location(
    "torchao_utils_for_test",
    SGLANG_ROOT / "sglang" / "srt" / "layers" / "torchao_utils.py",
)
torchao_utils_module = importlib.util.module_from_spec(torchao_utils_spec)
assert torchao_utils_spec.loader is not None
fake_torch = types.SimpleNamespace(nn=types.SimpleNamespace(Module=object))
with mock.patch.dict(sys.modules, {"torch": fake_torch}):
    torchao_utils_spec.loader.exec_module(torchao_utils_module)
apply_torchao_config_to_model = torchao_utils_module.apply_torchao_config_to_model


def write_fake_distribution(site_dir: Path, distribution_name: str, import_name: str, version: str) -> None:
    package_dir = site_dir / import_name
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(f"__version__ = {version!r}\n", encoding="utf-8")

    dist_info_dir = site_dir / f"{distribution_name.replace('-', '_')}-{version}.dist-info"
    dist_info_dir.mkdir(parents=True, exist_ok=True)
    (dist_info_dir / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {distribution_name}\nVersion: {version}\n",
        encoding="utf-8",
    )
    (dist_info_dir / "top_level.txt").write_text(f"{import_name}\n", encoding="utf-8")


def write_fake_local_package(root_dir: Path, module_name: str) -> None:
    package_dir = root_dir / module_name
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text("__version__ = 'local'\n", encoding="utf-8")


class BootstrapRuntimeOverlayTests(unittest.TestCase):
    def test_default_import_targets_use_lightweight_math_verify_probe(self):
        self.assertIn("math_verify", DEFAULT_IMPORT_TARGETS)
        self.assertNotIn("verl.workers.reward_manager.hf_math_verify", DEFAULT_IMPORT_TARGETS)

    def test_managed_runtime_support_requirements_cover_tensordict_import_deps(self):
        self.assertIn("cloudpickle", MANAGED_RUNTIME_SUPPORT_REQUIREMENTS)
        self.assertIn("importlib_metadata", MANAGED_RUNTIME_SUPPORT_REQUIREMENTS)
        self.assertIn("packaging", MANAGED_RUNTIME_SUPPORT_REQUIREMENTS)
        self.assertIn("pyvers<0.2.0,>=0.1.0", MANAGED_RUNTIME_SUPPORT_REQUIREMENTS)

    def test_build_managed_runtime_specs_uses_repo_versions_and_indexes(self):
        runtime_specs = build_managed_runtime_specs()

        self.assertEqual(runtime_specs["torch"].expected_version, "2.7.1+cu128")
        self.assertEqual(runtime_specs["torch"].install_requirement, "torch==2.7.1")
        self.assertEqual(runtime_specs["torch"].index_url, "https://download.pytorch.org/whl/cu128")
        self.assertEqual(runtime_specs["sgl-kernel"].expected_version, "0.2.8")
        self.assertTrue(
            runtime_specs["flashinfer-python"].find_links.endswith("/cu128/torch2.7/flashinfer-python")
        )

    def test_collect_requirement_index_includes_sglang_runtime_dependency(self):
        requirement_index = collect_requirement_index()
        self.assertIn("pybase64", requirement_index)
        self.assertEqual(requirement_index["pybase64"], "pybase64")

    def test_collect_requirement_index_includes_ninja_runtime_tool(self):
        requirement_index = collect_requirement_index()
        self.assertIn("ninja", requirement_index)
        self.assertEqual(requirement_index["ninja"], "ninja")

    def test_runtime_executable_candidate_paths_prefer_overlay_bin_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            job_overlay_dir = tmp_root / "job-overlay"
            reusable_overlay_dir = tmp_root / "reusable-overlay"
            module_file = reusable_overlay_dir / "ninja" / "__init__.py"

            candidates = runtime_executable_candidate_paths(
                "ninja",
                job_overlay_dir,
                reusable_overlay_dirs=[reusable_overlay_dir],
                module_origin=str(module_file),
            )

            self.assertEqual(candidates[0], job_overlay_dir / "bin" / "ninja")
            self.assertEqual(candidates[1], reusable_overlay_dir / "bin" / "ninja")
            self.assertIn(reusable_overlay_dir / "ninja" / "data" / "bin" / "ninja", candidates)

    def test_collect_requirement_index_overrides_with_transformers_compatibility_pins(self):
        requirement_index = collect_requirement_index()
        self.assertEqual(requirement_index["huggingface_hub"], "huggingface-hub>=0.34.0,<1.0")
        self.assertEqual(requirement_index["datasets"], "datasets>=2.15.0")
        self.assertEqual(requirement_index["regex"], "regex!=2019.12.17")

    def test_candidate_requirements_uses_special_mapping(self):
        requirement_index = build_requirement_index(["pyzmq>=25.1.2", "pillow"])
        zmq_candidates = candidate_requirements_for_module("zmq.green", requirement_index)
        pil_candidates = candidate_requirements_for_module("PIL.Image", requirement_index)

        self.assertEqual(zmq_candidates[0], "pyzmq>=25.1.2")
        self.assertEqual(pil_candidates[0], "pillow")

    def test_runtime_key_is_deterministic_and_root_is_versioned(self):
        runtime_specs = build_managed_runtime_specs()
        python_version = "3.10.20"
        reversed_specs = {name: runtime_specs[name] for name in reversed(list(runtime_specs))}

        runtime_key_one = build_runtime_key(python_version, runtime_specs)
        runtime_key_two = build_runtime_key(python_version, reversed_specs)
        runtime_root = select_runtime_root(Path("/tmp/runtime-envs"), runtime_key_one, python_version, runtime_specs)

        self.assertEqual(runtime_key_one, runtime_key_two)
        self.assertIn(runtime_key_one, runtime_root.name)
        self.assertIn("torch2_7_1_cu128", runtime_root.name)

    def test_lightweight_overlay_key_is_deterministic_and_versioned(self):
        overlay_key_one = build_lightweight_overlay_key(
            runtime_key="abc123",
            constraint_requirements=["datasets>=2.15.0", "python-multipart"],
            import_targets=["transformers", "sglang.srt.entrypoints.http_server"],
        )
        overlay_key_two = build_lightweight_overlay_key(
            runtime_key="abc123",
            constraint_requirements=["python-multipart", "datasets>=2.15.0"],
            import_targets=["transformers", "sglang.srt.entrypoints.http_server"],
        )
        overlay_dir = select_lightweight_overlay_dir(Path("/tmp/runtime-envs"), overlay_key_one)

        self.assertEqual(overlay_key_one, overlay_key_two)
        self.assertIn(overlay_key_one, str(overlay_dir))

    def test_validate_manifest_payload_detects_version_drift(self):
        runtime_spec = RuntimePackageSpec(
            distribution_name="torch",
            import_name="torch",
            expected_version="2.7.1+cu128",
            install_requirement="torch==2.7.1",
            index_url="https://download.pytorch.org/whl/cu128",
        )
        runtime_site = Path("/tmp/runtime-env/site-packages")
        manifest_payload = {
            "runtime_key": "abc123",
            "runtime_site_packages": str(runtime_site),
            "python_version": "3.10.20",
            "requirements": {
                "torch": {
                    "distribution_name": "torch",
                    "import_name": "torch",
                    "expected_version": "2.11.0",
                    "install_requirement": "torch==2.11.0",
                }
            },
        }

        reasons = validate_manifest_payload(
            manifest_payload=manifest_payload,
            runtime_key="abc123",
            runtime_site=runtime_site,
            runtime_specs={"torch": runtime_spec},
            python_version="3.10.20",
        )

        self.assertTrue(any("version mismatch" in reason for reason in reasons))

    def test_validate_managed_runtime_environment_accepts_clean_runtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            runtime_spec = RuntimePackageSpec(
                distribution_name="torch",
                import_name="torch",
                expected_version="2.7.1+cu128",
                install_requirement="torch==2.7.1",
                index_url="https://download.pytorch.org/whl/cu128",
            )
            runtime_specs = {"torch": runtime_spec}
            python_version = "3.10.20"
            local_import_modules = {
                "sglang": tmp_root / "local-sglang",
                "verl": tmp_root / "local-verl",
                "transformers": tmp_root / "local-transformers",
            }

            for module_name, root_dir in local_import_modules.items():
                write_fake_local_package(root_dir, module_name)

            runtime_key = build_runtime_key(python_version, runtime_specs, list(local_import_modules.values()))
            runtime_root = select_runtime_root(tmp_root / "runtime-envs", runtime_key, python_version, runtime_specs)
            runtime_site = runtime_site_packages_dir(runtime_root)
            runtime_site.mkdir(parents=True, exist_ok=True)
            job_overlay_dir = tmp_root / "job-overlay"
            job_overlay_dir.mkdir(parents=True, exist_ok=True)
            write_fake_distribution(runtime_site, "torch", "torch", "2.7.1+cu128")

            manifest_payload = {
                "runtime_key": runtime_key,
                "runtime_site_packages": str(runtime_site.resolve()),
                "python_version": python_version,
                "requirements": {"torch": asdict(runtime_spec)},
            }
            manifest_path_value = runtime_manifest_path(runtime_root)
            manifest_path_value.write_text(json.dumps(manifest_payload), encoding="utf-8")

            result = validate_managed_runtime_environment(
                python_executable=sys.executable,
                runtime_root=runtime_root,
                runtime_site=runtime_site,
                job_overlay_dir=job_overlay_dir,
                reusable_overlay_dirs=[],
                runtime_specs=runtime_specs,
                runtime_key=runtime_key,
                python_version=python_version,
                manifest_path_value=manifest_path_value,
                log=lambda _: None,
                local_import_modules=local_import_modules,
            )

            self.assertTrue(result.success)
            self.assertEqual(result.managed_packages["torch"].version, "2.7.1+cu128")
            self.assertIn("sglang", result.local_imports)
            self.assertEqual(result.sys_path_head[0], str(runtime_site.resolve()))
            self.assertEqual(result.sys_path_head[1], str(job_overlay_dir.resolve()))

    def test_validate_managed_runtime_environment_rejects_torch_from_job_overlay(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            runtime_spec = RuntimePackageSpec(
                distribution_name="torch",
                import_name="torch",
                expected_version="2.7.1+cu128",
                install_requirement="torch==2.7.1",
                index_url="https://download.pytorch.org/whl/cu128",
            )
            runtime_specs = {"torch": runtime_spec}
            python_version = "3.10.20"
            local_import_modules = {
                "sglang": tmp_root / "local-sglang",
                "verl": tmp_root / "local-verl",
                "transformers": tmp_root / "local-transformers",
            }

            for module_name, root_dir in local_import_modules.items():
                write_fake_local_package(root_dir, module_name)

            runtime_key = build_runtime_key(python_version, runtime_specs, list(local_import_modules.values()))
            runtime_root = select_runtime_root(tmp_root / "runtime-envs", runtime_key, python_version, runtime_specs)
            runtime_site = runtime_site_packages_dir(runtime_root)
            runtime_site.mkdir(parents=True, exist_ok=True)
            job_overlay_dir = tmp_root / "job-overlay"
            job_overlay_dir.mkdir(parents=True, exist_ok=True)
            write_fake_distribution(job_overlay_dir, "torch", "torch", "2.11.0")

            manifest_payload = {
                "runtime_key": runtime_key,
                "runtime_site_packages": str(runtime_site.resolve()),
                "python_version": python_version,
                "requirements": {"torch": asdict(runtime_spec)},
            }
            manifest_path_value = runtime_manifest_path(runtime_root)
            manifest_path_value.write_text(json.dumps(manifest_payload), encoding="utf-8")

            result = validate_managed_runtime_environment(
                python_executable=sys.executable,
                runtime_root=runtime_root,
                runtime_site=runtime_site,
                job_overlay_dir=job_overlay_dir,
                reusable_overlay_dirs=[],
                runtime_specs=runtime_specs,
                runtime_key=runtime_key,
                python_version=python_version,
                manifest_path_value=manifest_path_value,
                log=lambda _: None,
                local_import_modules=local_import_modules,
            )

            self.assertFalse(result.success)
            self.assertTrue(any("outside runtime site" in reason for reason in result.reasons))

    def test_probe_import_targets_runs_in_isolated_subprocess(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            runtime_site = tmp_root / "runtime-site"
            runtime_site.mkdir(parents=True, exist_ok=True)
            job_overlay_dir = tmp_root / "job-overlay"
            job_overlay_dir.mkdir(parents=True, exist_ok=True)
            local_import_root = tmp_root / "local-imports"
            local_import_root.mkdir(parents=True, exist_ok=True)
            write_fake_local_package(local_import_root, "toy_pkg")
            (local_import_root / "toy_pkg" / "module.py").write_text(
                "import helper_dep\nVALUE = 'ok'\n",
                encoding="utf-8",
            )

            initial = probe_import_targets(
                python_executable=sys.executable,
                runtime_site=runtime_site,
                job_overlay_dir=job_overlay_dir,
                import_targets=["toy_pkg.module"],
                reusable_overlay_dirs=[],
                local_import_paths=[local_import_root],
            )
            self.assertFalse(initial.success)
            self.assertEqual(initial.missing_module, "helper_dep")

            helper_package = job_overlay_dir / "helper_dep"
            helper_package.mkdir(parents=True, exist_ok=True)
            (helper_package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")

            repaired = probe_import_targets(
                python_executable=sys.executable,
                runtime_site=runtime_site,
                job_overlay_dir=job_overlay_dir,
                import_targets=["toy_pkg.module"],
                reusable_overlay_dirs=[],
                local_import_paths=[local_import_root],
            )
            self.assertTrue(repaired.success)

    def test_probe_import_targets_uses_reusable_overlay_dir_without_hydration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            runtime_site = tmp_root / "runtime-site"
            runtime_site.mkdir(parents=True, exist_ok=True)
            job_overlay_dir = tmp_root / "job-overlay"
            job_overlay_dir.mkdir(parents=True, exist_ok=True)
            reusable_overlay_dir = tmp_root / "reusable-overlay"
            reusable_overlay_dir.mkdir(parents=True, exist_ok=True)
            local_import_root = tmp_root / "local-imports"
            local_import_root.mkdir(parents=True, exist_ok=True)
            write_fake_local_package(local_import_root, "toy_pkg")
            (local_import_root / "toy_pkg" / "module.py").write_text(
                "import helper_dep\nVALUE = 'ok'\n",
                encoding="utf-8",
            )

            helper_package = reusable_overlay_dir / "helper_dep"
            helper_package.mkdir(parents=True, exist_ok=True)
            (helper_package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")

            repaired = probe_import_targets(
                python_executable=sys.executable,
                runtime_site=runtime_site,
                job_overlay_dir=job_overlay_dir,
                reusable_overlay_dirs=[reusable_overlay_dir],
                import_targets=["toy_pkg.module"],
                local_import_paths=[local_import_root],
            )

            self.assertTrue(repaired.success)

    def test_probe_import_targets_reports_exact_target_for_timeout(self):
        with mock.patch(
            "bootstrap_runtime_overlay.run_isolated_helper",
            return_value={
                "success": False,
                "target": None,
                "missing_module": None,
                "traceback_text": "Isolated helper timed out after 300 seconds.",
            },
        ):
            result = probe_import_targets(
                python_executable=sys.executable,
                runtime_site=Path("/tmp/runtime-site"),
                job_overlay_dir=Path("/tmp/job-overlay"),
                reusable_overlay_dirs=[],
                import_targets=["math_verify"],
                local_import_paths=[],
            )

        self.assertFalse(result.success)
        self.assertEqual(result.target, "math_verify")
        self.assertIn("timed out", result.traceback_text)

    def test_infer_requirements_from_traceback_extracts_transformers_version_pin(self):
        requirement_index = collect_requirement_index()
        traceback_text = (
            "ImportError: huggingface-hub>=0.34.0,<1.0 is required for a normal functioning of this module, "
            "but found huggingface-hub==1.11.0."
        )

        inferred = infer_requirements_from_traceback(traceback_text, requirement_index)

        self.assertEqual(inferred, ["huggingface-hub>=0.34.0,<1.0"])

    def test_infer_requirements_from_traceback_extracts_nested_missing_module(self):
        requirement_index = collect_requirement_index()
        traceback_text = (
            "Traceback (most recent call last):\n"
            "  File \"...\", line 1, in <module>\n"
            "ModuleNotFoundError: No module named 'tokenizers'\n"
            "...\n"
            "ModuleNotFoundError: Could not import module 'AutoConfig'. Are this object's requirements defined correctly?\n"
        )

        inferred = infer_requirements_from_traceback(traceback_text, requirement_index)

        self.assertIn("tokenizers>=0.21,<0.22", inferred)

    def test_infer_requirements_from_traceback_extracts_python_multipart_runtime_hint(self):
        requirement_index = collect_requirement_index()
        traceback_text = (
            'RuntimeError: Form data requires "python-multipart" to be installed.\n'
            'You can install "python-multipart" with:\n\n'
            "pip install python-multipart\n"
        )

        inferred = infer_requirements_from_traceback(traceback_text, requirement_index)

        self.assertIn("python-multipart", inferred)

    def test_build_lightweight_constraint_requirements_includes_transformers_pin(self):
        requirement_index = collect_requirement_index()

        constraints = build_lightweight_constraint_requirements(requirement_index)

        self.assertIn("huggingface-hub>=0.34.0,<1.0", constraints)
        self.assertIn("datasets>=2.15.0", constraints)
        self.assertNotIn("torch==2.7.1", constraints)

    def test_overlay_entry_requirement_key_handles_dist_info_names(self):
        self.assertEqual(
            overlay_entry_requirement_key("python_multipart-0.0.20.dist-info"),
            "python_multipart",
        )
        self.assertEqual(overlay_entry_requirement_key("requests"), "requests")

    def test_should_cache_overlay_entry_rejects_blocked_runtime_packages(self):
        self.assertFalse(should_cache_overlay_entry("torch"))
        self.assertFalse(should_cache_overlay_entry("transformers-4.55.4.dist-info"))
        self.assertFalse(should_cache_overlay_entry("nvidia_cublas_cu13-1.0.0.dist-info"))
        self.assertTrue(should_cache_overlay_entry("python_multipart-0.0.20.dist-info"))

    def test_bootstrap_runtime_repairs_non_module_version_conflict(self):
        requirement_index = collect_requirement_index()
        probe_results = iter(
            [
                ProbeResult(
                    False,
                    "transformers",
                    None,
                    "ImportError: huggingface-hub>=0.34.0,<1.0 is required for a normal functioning of this module, "
                    "but found huggingface-hub==1.11.0.",
                ),
                ProbeResult(True, None, None, ""),
            ]
        )
        installed = []

        repaired = bootstrap_runtime(
            import_targets=["transformers"],
            requirement_index=requirement_index,
            install_one=lambda requirement: installed.append(requirement),
            probe_once=lambda _: next(probe_results),
            log=lambda _: None,
            max_rounds=2,
        )

        self.assertEqual(repaired, ["huggingface-hub>=0.34.0,<1.0"])
        self.assertEqual(installed, ["huggingface-hub>=0.34.0,<1.0"])

    def test_bootstrap_runtime_repairs_package_metadata_error(self):
        requirement_index = collect_requirement_index()
        probe_results = iter(
            [
                ProbeResult(
                    False,
                    "transformers",
                    "The 'safetensors>=0.4.3' distribution was not found and is required by this application.",
                    "Traceback...\nimportlib.metadata.PackageNotFoundError: No package metadata was found for safetensors\n",
                ),
                ProbeResult(True, None, None, ""),
            ]
        )
        installed = []

        repaired = bootstrap_runtime(
            import_targets=["transformers"],
            requirement_index=requirement_index,
            install_one=lambda requirement: installed.append(requirement),
            probe_once=lambda _: next(probe_results),
            log=lambda _: None,
            max_rounds=2,
        )

        self.assertEqual(repaired, ["safetensors>=0.4.3"])
        self.assertEqual(installed, ["safetensors>=0.4.3"])

    def test_bootstrap_runtime_repairs_nested_missing_module_from_traceback(self):
        requirement_index = collect_requirement_index()
        probe_results = iter(
            [
                ProbeResult(
                    False,
                    "sglang.srt.entrypoints.http_server",
                    None,
                    "Traceback...\nModuleNotFoundError: No module named 'tokenizers'\n"
                    "ModuleNotFoundError: Could not import module 'AutoConfig'. Are this object's requirements defined correctly?\n",
                ),
                ProbeResult(True, None, None, ""),
            ]
        )
        installed = []

        repaired = bootstrap_runtime(
            import_targets=["sglang.srt.entrypoints.http_server"],
            requirement_index=requirement_index,
            install_one=lambda requirement: installed.append(requirement),
            probe_once=lambda _: next(probe_results),
            log=lambda _: None,
            max_rounds=2,
        )

        self.assertEqual(repaired, ["tokenizers>=0.21,<0.22"])
        self.assertEqual(installed, ["tokenizers>=0.21,<0.22"])

    def test_bootstrap_runtime_repairs_multiple_missing_modules(self):
        requirement_index = build_requirement_index(["pybase64", "orjson"])
        probe_results = iter(
            [
                ProbeResult(False, "sglang.srt.entrypoints.http_server", "pybase64", "traceback-1"),
                ProbeResult(False, "sglang.srt.entrypoints.http_server", "orjson", "traceback-2"),
                ProbeResult(True, None, None, ""),
            ]
        )
        installed = []
        logs = []

        repaired = bootstrap_runtime(
            import_targets=["sglang.srt.entrypoints.http_server"],
            requirement_index=requirement_index,
            install_one=lambda requirement: installed.append(requirement),
            probe_once=lambda _: next(probe_results),
            log=logs.append,
            max_rounds=4,
        )

        self.assertEqual(repaired, ["pybase64", "orjson"])
        self.assertEqual(installed, ["pybase64", "orjson"])
        self.assertTrue(any("pybase64" in message for message in logs))
        self.assertTrue(any("runtime import preflight succeeded" in message for message in logs))

    def test_bootstrap_runtime_succeeds_when_last_round_install_repairs_imports(self):
        requirement_index = build_requirement_index(["fastapi"])
        probe_results = iter(
            [
                ProbeResult(False, "sglang.srt.entrypoints.http_server", "fastapi", "traceback-fastapi"),
                ProbeResult(True, None, None, ""),
            ]
        )
        installed = []

        repaired = bootstrap_runtime(
            import_targets=["sglang.srt.entrypoints.http_server"],
            requirement_index=requirement_index,
            install_one=lambda requirement: installed.append(requirement),
            probe_once=lambda _: next(probe_results),
            log=lambda _: None,
            max_rounds=1,
        )

        self.assertEqual(repaired, ["fastapi"])
        self.assertEqual(installed, ["fastapi"])

    def test_bootstrap_runtime_retries_alternate_candidate_after_install_failure(self):
        probe_results = iter(
            [
                ProbeResult(False, "custom.module", "foo_bar", "traceback-1"),
                ProbeResult(True, None, None, ""),
            ]
        )
        attempted = []

        def install_one(requirement: str) -> None:
            attempted.append(requirement)
            if requirement == "foo_bar":
                raise RuntimeError("invalid package name")

        repaired = bootstrap_runtime(
            import_targets=["custom.module"],
            requirement_index={},
            install_one=install_one,
            probe_once=lambda _: next(probe_results),
            log=lambda _: None,
            max_rounds=2,
        )

        self.assertEqual(repaired, ["foo-bar"])
        self.assertEqual(attempted, ["foo_bar", "foo-bar"])

    def test_bootstrap_runtime_raises_for_non_module_error(self):
        with self.assertRaisesRegex(RuntimeError, "dependency/version/metadata error"):
            bootstrap_runtime(
                import_targets=["broken.module"],
                requirement_index={},
                install_one=lambda requirement: None,
                probe_once=lambda _: ProbeResult(False, "broken.module", None, "boom"),
                log=lambda _: None,
                max_rounds=1,
            )

    def test_bootstrap_runtime_exhaustion_reports_last_failure(self):
        with self.assertRaisesRegex(RuntimeError, "Last failing target: 'broken.module'"):
            bootstrap_runtime(
                import_targets=["broken.module"],
                requirement_index=build_requirement_index(["foo"]),
                install_one=lambda requirement: None,
                probe_once=lambda _: ProbeResult(False, "broken.module", "foo", "traceback-final"),
                log=lambda _: None,
                max_rounds=1,
            )

    def test_sync_overlay_entries_skips_blocked_runtime_packages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            source_dir = tmp_root / "source"
            target_dir = tmp_root / "target"
            source_dir.mkdir(parents=True, exist_ok=True)

            write_fake_distribution(source_dir, "requests", "requests", "2.33.1")
            write_fake_distribution(source_dir, "transformers", "transformers", "4.55.4")
            write_fake_distribution(source_dir, "nvidia-cublas-cu13", "nvidia_cublas_cu13", "1.0.0")

            copied_count = sync_overlay_entries(
                source_dir,
                target_dir,
                log=lambda _: None,
                context_label="[test]",
            )

            self.assertEqual(copied_count, 2)
            self.assertTrue((target_dir / "requests").exists())
            self.assertTrue(any(path.name.startswith("requests-") for path in target_dir.iterdir()))
            self.assertFalse((target_dir / "transformers").exists())
            self.assertFalse((target_dir / "nvidia_cublas_cu13").exists())

    def test_snapshot_and_changed_overlay_entries_detect_incremental_updates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            overlay_dir = Path(tmpdir)
            before_snapshot = snapshot_overlay_entries(overlay_dir)
            write_fake_distribution(overlay_dir, "requests", "requests", "2.33.1")

            changed_names = changed_overlay_entry_names(overlay_dir, before_snapshot)

            self.assertIn("requests", changed_names)
            self.assertTrue(any(name.startswith("requests-") for name in changed_names))

    def test_sync_overlay_entry_names_copies_only_selected_safe_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            source_dir = tmp_root / "source"
            target_dir = tmp_root / "target"
            source_dir.mkdir(parents=True, exist_ok=True)

            write_fake_distribution(source_dir, "requests", "requests", "2.33.1")
            write_fake_distribution(source_dir, "transformers", "transformers", "4.55.4")

            copied_count = sync_overlay_entry_names(
                source_dir,
                target_dir,
                ["requests", "requests-2.33.1.dist-info", "transformers", "transformers-4.55.4.dist-info"],
                log=lambda _: None,
                context_label="[test]",
            )

            self.assertEqual(copied_count, 2)
            self.assertTrue((target_dir / "requests").exists())
            self.assertFalse((target_dir / "transformers").exists())

    def test_sync_overlay_entries_skips_large_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            source_dir = tmp_root / "source"
            target_dir = tmp_root / "target"
            source_dir.mkdir(parents=True, exist_ok=True)

            small_entry = source_dir / "smallpkg"
            small_entry.mkdir(parents=True, exist_ok=True)
            (small_entry / "__init__.py").write_text("x = 1\n", encoding="utf-8")

            large_entry = source_dir / "largepkg"
            large_entry.mkdir(parents=True, exist_ok=True)
            (large_entry / "payload.bin").write_bytes(b"x" * 32)

            logs = []
            copied_count = sync_overlay_entries(
                source_dir,
                target_dir,
                log=logs.append,
                context_label="[test]",
                max_copy_bytes=16,
            )

            self.assertEqual(copied_count, 1)
            self.assertTrue((target_dir / "smallpkg").exists())
            self.assertFalse((target_dir / "largepkg").exists())
            self.assertTrue(any("skipped large overlay entry largepkg" in message for message in logs))

    def test_sync_overlay_entry_names_skips_large_entries(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            source_dir = tmp_root / "source"
            target_dir = tmp_root / "target"
            source_dir.mkdir(parents=True, exist_ok=True)

            small_entry = source_dir / "smallpkg"
            small_entry.mkdir(parents=True, exist_ok=True)
            (small_entry / "__init__.py").write_text("x = 1\n", encoding="utf-8")

            large_entry = source_dir / "largepkg"
            large_entry.mkdir(parents=True, exist_ok=True)
            (large_entry / "payload.bin").write_bytes(b"x" * 32)

            logs = []
            copied_count = sync_overlay_entry_names(
                source_dir,
                target_dir,
                ["smallpkg", "largepkg"],
                log=logs.append,
                context_label="[test]",
                max_copy_bytes=16,
            )

            self.assertEqual(copied_count, 1)
            self.assertTrue((target_dir / "smallpkg").exists())
            self.assertFalse((target_dir / "largepkg").exists())
            self.assertTrue(any("skipped large overlay entry largepkg" in message for message in logs))

    def test_purge_blocked_overlay_entries_removes_shadowing_packages(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            overlay_dir = Path(tmpdir)
            write_fake_distribution(overlay_dir, "requests", "requests", "2.33.1")
            write_fake_distribution(overlay_dir, "torch", "torch", "2.11.0")

            removed_count = purge_blocked_overlay_entries(
                overlay_dir,
                log=lambda _: None,
                context_label="[test]",
            )

            self.assertGreaterEqual(removed_count, 2)
            self.assertTrue((overlay_dir / "requests").exists())
            self.assertFalse((overlay_dir / "torch").exists())

    def test_install_requirement_uses_no_deps_for_compressed_tensors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            overlay_dir = Path(tmpdir)
            recorded_commands = []

            def fake_run(command, check):
                recorded_commands.append(command)

            with mock.patch("bootstrap_runtime_overlay.subprocess.run", side_effect=fake_run):
                install_requirement(
                    python_executable=sys.executable,
                    overlay_dir=overlay_dir,
                    requirement="compressed_tensors",
                    log=lambda _: None,
                )

            self.assertEqual(len(recorded_commands), 1)
            self.assertIn("--no-deps", recorded_commands[0])

    def test_install_requirement_refuses_heavyweight_torch_overlay(self):
        with self.assertRaisesRegex(RuntimeError, "refusing to install heavyweight runtime requirement"):
            install_requirement(
                python_executable=sys.executable,
                overlay_dir=Path.cwd(),
                requirement="torch",
                log=lambda _: None,
            )

    def test_safe_cpu_has_amx_support_returns_false_when_attr_missing(self):
        class FakeTorch:
            class _C:
                class _cpu:
                    pass

        self.assertFalse(safe_cpu_has_amx_support(FakeTorch, True))
        self.assertFalse(safe_cpu_has_amx_support(FakeTorch, False))

    def test_apply_torchao_config_to_model_skips_optional_import_for_empty_config(self):
        sentinel_model = object()

        result = apply_torchao_config_to_model(sentinel_model, "")

        self.assertIs(result, sentinel_model)

    def test_apply_torchao_config_to_model_raises_clear_error_when_torchao_missing(self):
        sentinel_model = object()

        with self.assertRaisesRegex(RuntimeError, "torchao_config was requested"):
            apply_torchao_config_to_model(sentinel_model, "int8wo")

    def test_build_isolated_pythonpath_orders_runtime_overlay_and_local_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            runtime_dir = tmp_root / "runtime"
            overlay_dir = tmp_root / "overlay"
            reusable_dir = tmp_root / "reusable"
            local_a = tmp_root / "local-a"
            local_b = tmp_root / "local-b"
            for path in (runtime_dir, overlay_dir, reusable_dir, local_a, local_b):
                path.mkdir(parents=True, exist_ok=True)

            pythonpath = build_isolated_pythonpath(
                runtime_dir,
                overlay_dir,
                [reusable_dir],
                [local_a, local_b],
            )
            self.assertEqual(
                pythonpath.split(":"),
                [
                    str(runtime_dir),
                    str(overlay_dir),
                    str(reusable_dir),
                    str(local_a),
                    str(local_b),
                ],
            )


if __name__ == "__main__":
    unittest.main()
