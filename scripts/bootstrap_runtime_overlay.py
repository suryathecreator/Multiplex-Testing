#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import importlib
from importlib import metadata as importlib_metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_IMPORT_MODULES = {
    "sglang": REPO_ROOT / "sglang-0.4.9.post6",
    "verl": REPO_ROOT / "verl-latest",
    "transformers": REPO_ROOT / "transformers-4.54.0" / "src",
}
LOCAL_IMPORT_PATHS = list(LOCAL_IMPORT_MODULES.values())
RUNTIME_EXECUTABLE_REQUIREMENTS = {
    "ninja": {
        "requirement": "ninja",
        "module_name": "ninja",
    },
}
DEFAULT_IMPORT_TARGETS = [
    "requests",
    "datasets",
    "matplotlib",
    "transformers",
    "sglang.srt.entrypoints.http_server",
    "sglang.srt.managers.tokenizer_manager",
    # Keep eval preflight lightweight; the driver scores directly with math_verify.
    "math_verify",
]

LIGHTWEIGHT_STATIC_REQUIREMENTS = [
    "requests",
    "datasets",
    "matplotlib==3.10.6",
    "pynvml==12.0.0",
    "latex2sympy2",
    "word2number",
    "pebble",
    "ninja",
    "timeout-decorator",
    "math-verify[antlr4_9_3]",
    "antlr4-python3-runtime==4.9.3",
    "addftool",
    "jsonlines",
    "math_verify",
    "tensorboardX",
    "aiohttp",
    "fastapi",
    "huggingface_hub",
    "msgspec",
    "orjson",
    "packaging",
    "partial_json_parser",
    "pillow",
    "prometheus-client>=0.20.0",
    "psutil",
    "pydantic",
    "pybase64",
    "python-multipart",
    "pyzmq>=25.1.2",
    "sentencepiece",
    "soundfile==0.13.1",
    "scipy",
    "uvicorn",
    "uvloop",
]

TRANSFORMERS_COMPATIBILITY_KEYS = [
    "Pillow",
    "datasets",
    "filelock",
    "fsspec",
    "huggingface-hub",
    "numpy",
    "packaging",
    "pandas",
    "pyyaml",
    "regex",
    "requests",
    "safetensors",
    "tokenizers",
    "tqdm",
    "urllib3",
]

SPECIAL_MODULE_REQUIREMENT_KEYS = {
    "pil": "pillow",
    "yaml": "pyyaml",
    "zmq": "pyzmq",
    "multipart": "python_multipart",
    "prometheus_client": "prometheus_client",
    "dateutil": "python_dateutil",
    "antlr4": "antlr4_python3_runtime",
}

BLOCKED_REQUIREMENT_KEYS = {
    "flashinfer",
    "flashinfer_python",
    "sgl_kernel",
    "sglang",
    "tensordict",
    "torch",
    "torchaudio",
    "torchvision",
    "transformers",
    "triton",
    "verl",
    "vllm",
}

COPYABLE_RUNTIME_DISTRIBUTIONS = {
    "flashinfer-python",
    "sgl-kernel",
    "tensordict",
    "triton",
}

MANAGED_RUNTIME_SUPPORT_REQUIREMENTS = [
    "cloudpickle",
    "importlib_metadata",
    "packaging",
    "pyvers<0.2.0,>=0.1.0",
]

MANAGED_RUNTIME_FALLBACK_VERSIONS = {
    "flashinfer-python": "0.2.9rc2",
    "sgl-kernel": "0.2.8",
    "tensordict": "0.10.0",
    "torch": "2.7.1+cu128",
    "torchaudio": "2.7.1+cu128",
    "torchvision": "0.22.1+cu128",
    "triton": "3.3.1",
}

LEGACY_OVERLAY_NAME = "python-overlay"
PYTORCH_INDEX_TEMPLATE = "https://download.pytorch.org/whl/{cuda_tag}"
FLASHINFER_FIND_LINKS_TEMPLATE = (
    "https://flashinfer.ai/whl/{cuda_tag}/torch{torch_major_minor}/flashinfer-python"
)
REQUIREMENT_SPLIT_RE = re.compile(r"[<>=!~;\[]")
CONDA_ENV_DOUBLE_EQUALS_RE = re.compile(r"^\s*-\s*([A-Za-z0-9_.-]+)==([^\s#]+)")
CONDA_ENV_SINGLE_EQUALS_RE = re.compile(r"^\s*-\s*([A-Za-z0-9_.-]+)=([^\s=#]+)")
IMPORT_ERROR_REQUIRED_RE = re.compile(r"ImportError:\s+(.+?)\s+is required\b")
PACKAGE_METADATA_MISSING_RE = re.compile(r"No package metadata was found for ([A-Za-z0-9_.-]+)")
NESTED_MODULE_NOT_FOUND_RE = re.compile(r"No module named '([A-Za-z0-9_.-]+)'")
RUNTIME_REQUIRES_PACKAGE_RE = re.compile(
    r'requires\s+"?([A-Za-z0-9_.-]+)"?\s+to be installed',
    re.IGNORECASE,
)
PIP_INSTALL_HINT_RE = re.compile(r"\bpip(?:3)?\s+install\s+([A-Za-z0-9_.-]+(?:\[[^\]]+\])?)")
MODULE_LIKE_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
NON_INSTALLABLE_REQUIREMENT_KEYS = {"python"}
ISOLATED_HELPER_TIMEOUT_SECONDS = 300
DIST_INFO_NAME_RE = re.compile(r"^(?P<name>.+?)-\d")
LIGHTWEIGHT_OVERLAY_DIRNAME = "lightweight-overlays"
OVERLAY_NO_DEPS_REQUIREMENT_KEYS = {"compressed_tensors"}
BLOCKED_OVERLAY_PREFIX_KEYS = ("cuda", "nvidia")
OVERLAY_CACHE_MAX_COPY_BYTES = 64 * 1024 * 1024
ISOLATED_HELPER_CODE = r"""
import importlib
import importlib.util
import json
import os
import sys
import traceback
from importlib import metadata as importlib_metadata
from pathlib import Path


def module_origin(module):
    origin = getattr(module, "__file__", None)
    if origin:
        return str(Path(origin).resolve())

    spec = getattr(module, "__spec__", None)
    if spec is None:
        return None
    if spec.origin not in (None, "built-in"):
        return str(Path(spec.origin).resolve())
    if spec.submodule_search_locations:
        return str(Path(next(iter(spec.submodule_search_locations))).resolve())
    return None


def inspect_local_import(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        raise ModuleNotFoundError(module_name)

    if spec.origin not in (None, "built-in"):
        return str(Path(spec.origin).resolve())
    if spec.submodule_search_locations:
        return str(Path(next(iter(spec.submodule_search_locations))).resolve())
    raise RuntimeError(f"Unable to determine import origin for {module_name!r}")


payload = json.loads(os.environ["BOOTSTRAP_HELPER_PAYLOAD"])
result = {
    "success": False,
    "sys_path_head": [entry for entry in sys.path if entry][:12],
}

try:
    kind = payload["kind"]
    if kind == "package":
        module = importlib.import_module(payload["import_name"])
        origin = module_origin(module)
        try:
            distribution = importlib_metadata.distribution(payload["distribution_name"])
            version = distribution.version
            distribution_path = getattr(distribution, "_path", None)
            dist_info_path = str(Path(distribution_path).resolve()) if distribution_path else None
        except importlib_metadata.PackageNotFoundError:
            version = getattr(module, "__version__", None)
            dist_info_path = None

        result.update(
            {
                "success": True,
                "origin": origin,
                "version": version,
                "dist_info_path": dist_info_path,
            }
        )
    elif kind == "local_imports":
        origins = {}
        for module_name in payload["module_names"]:
            origins[module_name] = inspect_local_import(module_name)
        result.update({"success": True, "origins": origins})
    elif kind == "probe_targets":
        for target in payload["import_targets"]:
            try:
                importlib.import_module(target)
            except ModuleNotFoundError as exc:
                result.update(
                    {
                        "target": target,
                        "missing_module": exc.name,
                        "traceback_text": traceback.format_exc(),
                    }
                )
                break
            except Exception:
                result.update(
                    {
                        "target": target,
                        "missing_module": None,
                        "traceback_text": traceback.format_exc(),
                    }
                )
                break
        else:
            result.update(
                {
                    "success": True,
                    "target": None,
                    "missing_module": None,
                    "traceback_text": "",
                }
            )
    else:
        raise RuntimeError(f"Unsupported helper payload kind: {kind!r}")
except Exception:
    result["traceback_text"] = traceback.format_exc()

print(json.dumps(result))
"""


@dataclass(frozen=True)
class RuntimePackageSpec:
    distribution_name: str
    import_name: str
    expected_version: str
    install_requirement: str
    index_url: Optional[str] = None
    find_links: Optional[str] = None
    no_deps: bool = False


@dataclass
class ProbeResult:
    success: bool
    target: Optional[str]
    missing_module: Optional[str]
    traceback_text: str


@dataclass
class ImportState:
    import_name: str
    origin: Optional[str]
    version: Optional[str]
    dist_info_path: Optional[str]


@dataclass
class RuntimeValidationResult:
    success: bool
    reasons: List[str]
    managed_packages: Dict[str, ImportState]
    local_imports: Dict[str, str]
    sys_path_head: List[str]


def normalize_requirement_key(requirement: str) -> str:
    root = REQUIREMENT_SPLIT_RE.split(requirement, maxsplit=1)[0].strip()
    return root.replace("-", "_").lower()


def dedupe_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in items:
        if item and item not in seen:
            ordered.append(item)
            seen.add(item)
    return ordered


def load_pyproject_requirements(pyproject_path: Path) -> List[str]:
    if tomllib is None or not pyproject_path.exists():
        return []

    with pyproject_path.open("rb") as fp:
        data = tomllib.load(fp)

    project = data.get("project", {})
    requirements: List[str] = list(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {})
    requirements.extend(optional.get("runtime_common", []))
    return requirements


def extract_string_list(node: ast.AST, known_assignments: Dict[str, List[str]]) -> Optional[List[str]]:
    if isinstance(node, ast.List):
        values: List[str] = []
        for element in node.elts:
            if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
                return None
            values.append(element.value)
        return values

    if isinstance(node, ast.Name):
        referenced = known_assignments.get(node.id)
        if referenced is None:
            return None
        return list(referenced)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = extract_string_list(node.left, known_assignments)
        right = extract_string_list(node.right, known_assignments)
        if left is None or right is None:
            return None
        return left + right

    return None


def load_setup_py_requirements(setup_py_path: Path, variable_names: Sequence[str]) -> List[str]:
    if not setup_py_path.exists():
        return []

    tree = ast.parse(setup_py_path.read_text(encoding="utf-8"), filename=str(setup_py_path))
    assignments: Dict[str, List[str]] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        values = extract_string_list(node.value, assignments)
        if values is None:
            continue
        assignments[node.targets[0].id] = values

    requirements: List[str] = []
    for variable_name in variable_names:
        requirements.extend(assignments.get(variable_name, []))
    return requirements


def build_requirement_index(requirements: Sequence[str]) -> Dict[str, str]:
    index: Dict[str, str] = {}
    for requirement in requirements:
        key = normalize_requirement_key(requirement)
        index.setdefault(key, requirement)
    return index


def load_transformers_dependency_requirements(
    dependency_table_path: Path,
    dependency_keys: Sequence[str] = TRANSFORMERS_COMPATIBILITY_KEYS,
) -> Dict[str, str]:
    if not dependency_table_path.exists():
        return {}

    namespace: Dict[str, object] = {}
    exec(dependency_table_path.read_text(encoding="utf-8"), namespace)
    deps = namespace.get("deps", {})
    if not isinstance(deps, dict):
        return {}

    requirements: Dict[str, str] = {}
    for dependency_key in dependency_keys:
        requirement = deps.get(dependency_key)
        if isinstance(requirement, str):
            requirements[normalize_requirement_key(requirement)] = requirement
    return requirements


def collect_requirement_index() -> Dict[str, str]:
    requirements = list(LIGHTWEIGHT_STATIC_REQUIREMENTS)
    requirements.extend(load_pyproject_requirements(REPO_ROOT / "sglang-0.4.9.post6" / "pyproject.toml"))
    requirements.extend(load_setup_py_requirements(REPO_ROOT / "verl-latest" / "setup.py", ["install_requires"]))

    requirement_index = build_requirement_index(dedupe_preserve_order(requirements))
    transformer_constraints = load_transformers_dependency_requirements(
        REPO_ROOT / "transformers-4.54.0" / "src" / "transformers" / "dependency_versions_table.py"
    )
    requirement_index.update(transformer_constraints)
    return requirement_index


def candidate_requirements_for_module(
    module_name: str,
    requirement_index: Dict[str, str],
) -> List[str]:
    top_level = module_name.split(".", 1)[0]
    normalized = top_level.replace("-", "_").lower()

    candidates: List[str] = []

    special_key = SPECIAL_MODULE_REQUIREMENT_KEYS.get(normalized)
    if special_key:
        candidates.append(requirement_index.get(special_key, special_key.replace("_", "-")))

    if normalized in requirement_index:
        candidates.append(requirement_index[normalized])

    candidates.append(top_level)
    if "_" in top_level:
        candidates.append(top_level.replace("_", "-"))
    if "-" in top_level:
        candidates.append(top_level.replace("-", "_"))

    return dedupe_preserve_order(candidates)


def build_lightweight_constraint_requirements(requirement_index: Dict[str, str]) -> List[str]:
    constraints: List[str] = []
    for requirement in requirement_index.values():
        requirement_key = normalize_requirement_key(requirement)
        if requirement_key in BLOCKED_REQUIREMENT_KEYS or requirement_key in NON_INSTALLABLE_REQUIREMENT_KEYS:
            continue
        if "[" in requirement:
            continue
        constraints.append(requirement)
    return dedupe_preserve_order(constraints)


def infer_requirements_from_traceback(
    traceback_text: str,
    requirement_index: Dict[str, str],
) -> List[str]:
    candidates: List[str] = []

    required_match = IMPORT_ERROR_REQUIRED_RE.search(traceback_text)
    if required_match:
        required_requirement = required_match.group(1).strip()
        requirement_key = normalize_requirement_key(required_requirement)
        candidates.append(requirement_index.get(requirement_key, required_requirement))

    metadata_match = PACKAGE_METADATA_MISSING_RE.search(traceback_text)
    if metadata_match:
        missing_package = metadata_match.group(1)
        candidates.extend(candidate_requirements_for_module(missing_package, requirement_index))

    for module_name in NESTED_MODULE_NOT_FOUND_RE.findall(traceback_text):
        if MODULE_LIKE_NAME_RE.fullmatch(module_name):
            candidates.extend(candidate_requirements_for_module(module_name, requirement_index))

    for package_name in RUNTIME_REQUIRES_PACKAGE_RE.findall(traceback_text):
        if MODULE_LIKE_NAME_RE.fullmatch(package_name):
            candidates.extend(candidate_requirements_for_module(package_name, requirement_index))

    for package_hint in PIP_INSTALL_HINT_RE.findall(traceback_text):
        candidates.extend(candidate_requirements_for_module(package_hint, requirement_index))

    return dedupe_preserve_order(candidates)


def load_conda_env_versions(conda_env_path: Path) -> Dict[str, str]:
    versions: Dict[str, str] = {}
    if not conda_env_path.exists():
        return versions

    for line in conda_env_path.read_text(encoding="utf-8").splitlines():
        match = CONDA_ENV_DOUBLE_EQUALS_RE.match(line)
        if match:
            versions.setdefault(normalize_requirement_key(match.group(1)), match.group(2))
            continue

        match = CONDA_ENV_SINGLE_EQUALS_RE.match(line)
        if match:
            versions.setdefault(normalize_requirement_key(match.group(1)), match.group(2))

    return versions


def public_version(version_string: str) -> str:
    return version_string.split("+", 1)[0]


def infer_cuda_tag(torch_version: str) -> str:
    if "+" in torch_version:
        return torch_version.split("+", 1)[1]
    return "cpu"


def pytorch_index_url(cuda_tag: str) -> str:
    if cuda_tag == "cpu":
        return "https://download.pytorch.org/whl/cpu"
    return PYTORCH_INDEX_TEMPLATE.format(cuda_tag=cuda_tag)


def build_managed_runtime_specs(
    conda_env_path: Path = REPO_ROOT / "conda_env.yaml",
) -> Dict[str, RuntimePackageSpec]:
    conda_versions = load_conda_env_versions(conda_env_path)

    def version_for(package_name: str) -> str:
        key = normalize_requirement_key(package_name)
        return conda_versions.get(key, MANAGED_RUNTIME_FALLBACK_VERSIONS[package_name])

    torch_expected = version_for("torch")
    torchvision_expected = version_for("torchvision")
    torchaudio_expected = version_for("torchaudio")
    cuda_tag = infer_cuda_tag(torch_expected)
    torch_major_minor = ".".join(public_version(torch_expected).split(".")[:2])
    torch_index_url = pytorch_index_url(cuda_tag)
    flashinfer_find_links = FLASHINFER_FIND_LINKS_TEMPLATE.format(
        cuda_tag=cuda_tag,
        torch_major_minor=torch_major_minor,
    )

    return {
        "torch": RuntimePackageSpec(
            distribution_name="torch",
            import_name="torch",
            expected_version=torch_expected,
            install_requirement=f"torch=={public_version(torch_expected)}",
            index_url=torch_index_url,
        ),
        "torchvision": RuntimePackageSpec(
            distribution_name="torchvision",
            import_name="torchvision",
            expected_version=torchvision_expected,
            install_requirement=f"torchvision=={public_version(torchvision_expected)}",
            index_url=torch_index_url,
        ),
        "torchaudio": RuntimePackageSpec(
            distribution_name="torchaudio",
            import_name="torchaudio",
            expected_version=torchaudio_expected,
            install_requirement=f"torchaudio=={public_version(torchaudio_expected)}",
            index_url=torch_index_url,
        ),
        "triton": RuntimePackageSpec(
            distribution_name="triton",
            import_name="triton",
            expected_version=version_for("triton"),
            install_requirement=f"triton=={public_version(version_for('triton'))}",
            no_deps=True,
        ),
        "tensordict": RuntimePackageSpec(
            distribution_name="tensordict",
            import_name="tensordict",
            expected_version=version_for("tensordict"),
            install_requirement=f"tensordict=={public_version(version_for('tensordict'))}",
            no_deps=True,
        ),
        "sgl-kernel": RuntimePackageSpec(
            distribution_name="sgl-kernel",
            import_name="sgl_kernel",
            expected_version=version_for("sgl-kernel"),
            install_requirement=f"sgl-kernel=={public_version(version_for('sgl-kernel'))}",
            no_deps=True,
        ),
        "flashinfer-python": RuntimePackageSpec(
            distribution_name="flashinfer-python",
            import_name="flashinfer",
            expected_version=version_for("flashinfer-python"),
            install_requirement=f"flashinfer-python=={public_version(version_for('flashinfer-python'))}",
            find_links=flashinfer_find_links,
            no_deps=True,
        ),
    }


def build_runtime_key(
    python_version: str,
    runtime_specs: Dict[str, RuntimePackageSpec],
    local_import_paths: Sequence[Path] = LOCAL_IMPORT_PATHS,
) -> str:
    payload = {
        "python_version": python_version,
        "requirements": {key: asdict(runtime_specs[key]) for key in sorted(runtime_specs)},
        "local_import_paths": [str(path.resolve()) for path in local_import_paths],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def build_lightweight_overlay_key(
    runtime_key: str,
    constraint_requirements: Sequence[str],
    import_targets: Sequence[str],
) -> str:
    payload = {
        "runtime_key": runtime_key,
        "constraint_requirements": sorted(constraint_requirements),
        "import_targets": list(import_targets),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]


def select_runtime_root(
    runtime_root_base: Path,
    runtime_key: str,
    python_version: str,
    runtime_specs: Dict[str, RuntimePackageSpec],
) -> Path:
    py_tag = "".join(python_version.split(".")[:2])
    torch_tag = runtime_specs["torch"].expected_version.replace("+", "_").replace(".", "_")
    return runtime_root_base / f"py{py_tag}-torch{torch_tag}-{runtime_key}"


def runtime_site_packages_dir(runtime_root: Path) -> Path:
    return runtime_root / "site-packages"


def runtime_manifest_path(runtime_root: Path) -> Path:
    return runtime_root / "runtime-manifest.json"


def select_lightweight_overlay_dir(runtime_root_base: Path, overlay_key: str) -> Path:
    return runtime_root_base.parent / LIGHTWEIGHT_OVERLAY_DIRNAME / overlay_key


def runtime_executable_candidate_paths(
    command_name: str,
    job_overlay_dir: Path,
    reusable_overlay_dirs: Sequence[Path] = (),
    module_origin: Optional[str] = None,
) -> List[Path]:
    candidates: List[Path] = [job_overlay_dir / "bin" / command_name]
    candidates.extend(Path(overlay_dir).resolve() / "bin" / command_name for overlay_dir in reusable_overlay_dirs)

    if module_origin:
        module_path = Path(module_origin).resolve()
        module_root = (
            module_path.parent
            if module_path.is_file() or module_path.suffix or module_path.name == "__init__.py"
            else module_path
        )
        candidates.extend(
            [
                module_root / "data" / "bin" / command_name,
                module_root / "bin" / command_name,
                module_root / command_name,
            ]
        )

    deduped: List[Path] = []
    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        deduped.append(candidate)
        seen.add(candidate_str)
    return deduped


def path_is_within(path_value: Optional[str], base_dir: Path) -> bool:
    if not path_value:
        return False
    try:
        return Path(path_value).resolve().is_relative_to(base_dir.resolve())
    except (OSError, RuntimeError, ValueError):
        return False


def build_isolated_pythonpath(
    runtime_site: Path,
    job_overlay_dir: Path,
    reusable_overlay_dirs: Sequence[Path] = (),
    local_import_paths: Sequence[Path] = LOCAL_IMPORT_PATHS,
) -> str:
    entries = [str(runtime_site.resolve()), str(job_overlay_dir.resolve())]
    entries.extend(str(path.resolve()) for path in reusable_overlay_dirs if path.exists())
    entries.extend(str(path.resolve()) for path in local_import_paths)
    return ":".join(entries)


def prepend_sys_path_entries(entries: Sequence[Path]) -> None:
    normalized = [str(entry.resolve()) for entry in entries if entry.exists()]
    for entry in reversed(normalized):
        while entry in sys.path:
            sys.path.remove(entry)
        sys.path.insert(0, entry)


def purge_module_prefixes(prefixes: Iterable[str]) -> None:
    prefix_list = [prefix for prefix in prefixes if prefix]
    for module_name in list(sys.modules):
        if any(module_name == prefix or module_name.startswith(f"{prefix}.") for prefix in prefix_list):
            sys.modules.pop(module_name, None)


def prepare_import_environment(
    runtime_site: Path,
    job_overlay_dir: Path,
    reusable_overlay_dirs: Sequence[Path] = (),
    local_import_paths: Sequence[Path] = LOCAL_IMPORT_PATHS,
) -> None:
    prepend_sys_path_entries([runtime_site, job_overlay_dir, *reusable_overlay_dirs, *local_import_paths])


def module_origin(module: object) -> Optional[str]:
    origin = getattr(module, "__file__", None)
    if origin:
        return str(Path(origin).resolve())

    spec = getattr(module, "__spec__", None)
    if spec is None:
        return None
    if spec.origin not in (None, "built-in"):
        return str(Path(spec.origin).resolve())
    if spec.submodule_search_locations:
        return str(Path(next(iter(spec.submodule_search_locations))).resolve())
    return None


def run_isolated_helper(
    *,
    python_executable: str,
    runtime_site: Path,
    job_overlay_dir: Path,
    payload: Dict[str, object],
    reusable_overlay_dirs: Sequence[Path] = (),
    local_import_paths: Sequence[Path] = LOCAL_IMPORT_PATHS,
    timeout_seconds: int = ISOLATED_HELPER_TIMEOUT_SECONDS,
) -> Dict[str, object]:
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONPATH"] = build_isolated_pythonpath(
        runtime_site,
        job_overlay_dir,
        reusable_overlay_dirs,
        local_import_paths,
    )
    env["BOOTSTRAP_HELPER_PAYLOAD"] = json.dumps(payload)

    timeout_target: Optional[str] = None
    if payload.get("kind") == "probe_targets":
        import_targets = payload.get("import_targets")
        if isinstance(import_targets, list) and len(import_targets) == 1 and isinstance(import_targets[0], str):
            timeout_target = import_targets[0]

    try:
        completed = subprocess.run(
            [python_executable, "-S", "-c", ISOLATED_HELPER_CODE],
            check=False,
            capture_output=True,
            env=env,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        return {
            "success": False,
            "target": timeout_target,
            "traceback_text": (
                "Isolated helper timed out after "
                f"{timeout_seconds} seconds.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
            ),
            "sys_path_head": env["PYTHONPATH"].split(":"),
        }

    stdout = (completed.stdout or "").strip()
    if not stdout:
        return {
            "success": False,
            "traceback_text": (
                "Isolated helper produced no JSON output.\n"
                f"returncode={completed.returncode}\nSTDERR:\n{completed.stderr}"
            ),
            "sys_path_head": env["PYTHONPATH"].split(":"),
        }

    json_line = stdout.splitlines()[-1]
    try:
        result = json.loads(json_line)
    except json.JSONDecodeError as exc:
        return {
            "success": False,
            "traceback_text": (
                "Isolated helper produced malformed JSON.\n"
                f"returncode={completed.returncode}\nSTDOUT:\n{stdout}\nSTDERR:\n{completed.stderr}\n"
                f"JSON error: {exc}"
            ),
            "sys_path_head": env["PYTHONPATH"].split(":"),
        }

    if completed.returncode != 0:
        result["success"] = False
        result["traceback_text"] = result.get("traceback_text") or (
            f"Isolated helper exited with code {completed.returncode}.\nSTDERR:\n{completed.stderr}"
        )

    if "sys_path_head" not in result:
        result["sys_path_head"] = env["PYTHONPATH"].split(":")
    return result


def load_manifest(manifest_path_value: Path) -> Optional[Dict[str, object]]:
    if not manifest_path_value.exists():
        return None
    return json.loads(manifest_path_value.read_text(encoding="utf-8"))


def validate_manifest_payload(
    manifest_payload: Optional[Dict[str, object]],
    runtime_key: str,
    runtime_site: Path,
    runtime_specs: Dict[str, RuntimePackageSpec],
    python_version: str,
) -> List[str]:
    if manifest_payload is None:
        return ["missing runtime manifest"]

    reasons: List[str] = []
    if manifest_payload.get("runtime_key") != runtime_key:
        reasons.append("runtime manifest key mismatch")
    if manifest_payload.get("runtime_site_packages") != str(runtime_site.resolve()):
        reasons.append("runtime manifest site-packages path mismatch")
    if manifest_payload.get("python_version") != python_version:
        reasons.append("runtime manifest python version mismatch")

    manifest_requirements = manifest_payload.get("requirements", {})
    if not isinstance(manifest_requirements, dict):
        return reasons + ["runtime manifest requirements section is malformed"]

    for distribution_name, spec in runtime_specs.items():
        entry = manifest_requirements.get(distribution_name)
        if not isinstance(entry, dict):
            reasons.append(f"runtime manifest missing requirement entry for {distribution_name}")
            continue
        if entry.get("expected_version") != spec.expected_version:
            reasons.append(f"runtime manifest version mismatch for {distribution_name}")
        if entry.get("install_requirement") != spec.install_requirement:
            reasons.append(f"runtime manifest install requirement mismatch for {distribution_name}")

    return reasons


def validate_managed_runtime_environment(
    *,
    python_executable: str,
    runtime_root: Path,
    runtime_site: Path,
    job_overlay_dir: Path,
    reusable_overlay_dirs: Sequence[Path],
    runtime_specs: Dict[str, RuntimePackageSpec],
    runtime_key: str,
    python_version: str,
    manifest_path_value: Path,
    log: Callable[[str], None],
    local_import_modules: Dict[str, Path] = LOCAL_IMPORT_MODULES,
    require_manifest: bool = True,
) -> RuntimeValidationResult:
    reasons: List[str] = []
    managed_packages: Dict[str, ImportState] = {}
    local_imports: Dict[str, str] = {}
    sys_path_head = build_isolated_pythonpath(
        runtime_site,
        job_overlay_dir,
        reusable_overlay_dirs,
        list(local_import_modules.values()),
    ).split(":")

    if not runtime_root.exists():
        reasons.append(f"managed runtime root missing: {runtime_root}")
    if not runtime_site.exists():
        reasons.append(f"managed runtime site-packages missing: {runtime_site}")

    manifest_payload = load_manifest(manifest_path_value)
    if require_manifest or manifest_payload is not None:
        reasons.extend(
            validate_manifest_payload(
                manifest_payload=manifest_payload,
                runtime_key=runtime_key,
                runtime_site=runtime_site,
                runtime_specs=runtime_specs,
                python_version=python_version,
            )
        )

    for distribution_name, spec in runtime_specs.items():
        helper_result = run_isolated_helper(
            python_executable=python_executable,
            runtime_site=runtime_site,
            job_overlay_dir=job_overlay_dir,
            reusable_overlay_dirs=reusable_overlay_dirs,
            local_import_paths=list(local_import_modules.values()),
            payload={
                "kind": "package",
                "distribution_name": spec.distribution_name,
                "import_name": spec.import_name,
            },
        )
        sys_path_head = list(helper_result.get("sys_path_head", sys_path_head))
        if not helper_result.get("success"):
            reasons.append(
                f"failed to import managed runtime package {distribution_name!r}:\n"
                f"{helper_result.get('traceback_text', 'unknown isolated helper failure')}"
            )
            continue
        state = ImportState(
            import_name=spec.import_name,
            origin=helper_result.get("origin"),
            version=helper_result.get("version"),
            dist_info_path=helper_result.get("dist_info_path"),
        )

        managed_packages[distribution_name] = state
        if not path_is_within(state.origin, runtime_site):
            reasons.append(
                f"managed runtime package {distribution_name!r} resolved outside runtime site: {state.origin}"
            )
        if not path_is_within(state.dist_info_path, runtime_site):
            reasons.append(
                f"managed runtime dist-info for {distribution_name!r} resolved outside runtime site: {state.dist_info_path}"
            )
        if state.version != spec.expected_version:
            reasons.append(
                f"managed runtime version mismatch for {distribution_name!r}: "
                f"expected {spec.expected_version!r}, got {state.version!r}"
            )

    local_import_result = run_isolated_helper(
        python_executable=python_executable,
        runtime_site=runtime_site,
        job_overlay_dir=job_overlay_dir,
        reusable_overlay_dirs=reusable_overlay_dirs,
        local_import_paths=list(local_import_modules.values()),
        payload={
            "kind": "local_imports",
            "module_names": list(local_import_modules),
        },
    )
    sys_path_head = list(local_import_result.get("sys_path_head", sys_path_head))
    if not local_import_result.get("success"):
        reasons.append(
            "[runtime] failed to resolve local imports in isolated validation process:\n"
            f"{local_import_result.get('traceback_text', 'unknown isolated helper failure')}"
        )
    else:
        resolved_origins = local_import_result.get("origins", {})
        if not isinstance(resolved_origins, dict):
            reasons.append("[runtime] isolated local import validation returned malformed origins payload")
        else:
            for module_name, module_root in local_import_modules.items():
                origin = resolved_origins.get(module_name)
                if not isinstance(origin, str):
                    reasons.append(f"failed to resolve local import {module_name!r}: missing origin")
                    continue
                local_imports[module_name] = origin
                if not path_is_within(origin, module_root):
                    reasons.append(f"local import {module_name!r} resolved outside patched repo path: {origin}")

    success = not reasons
    if success:
        log(f"[runtime] managed runtime validated successfully at {runtime_root}")
    else:
        log("[runtime] managed runtime validation failed")
        for reason in reasons:
            log(f"[runtime] reason={reason}")

    return RuntimeValidationResult(
        success=success,
        reasons=reasons,
        managed_packages=managed_packages,
        local_imports=local_imports,
        sys_path_head=sys_path_head,
    )


def copy_path_entry(source_path: Path, target_path: Path) -> None:
    if target_path.exists():
        if target_path.is_dir():
            shutil.rmtree(target_path)
        else:
            target_path.unlink()

    if source_path.is_dir():
        shutil.copytree(source_path, target_path)
    else:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def overlay_entry_size_bytes(path_value: Path) -> int:
    if not path_value.exists():
        return 0
    if path_value.is_file():
        return path_value.stat(follow_symlinks=False).st_size

    total_size = 0
    for root, _, files in os.walk(path_value):
        root_path = Path(root)
        for file_name in files:
            file_path = root_path / file_name
            try:
                total_size += file_path.stat(follow_symlinks=False).st_size
            except FileNotFoundError:
                continue
    return total_size


def maybe_copy_distribution_from_current_env(
    spec: RuntimePackageSpec,
    runtime_site: Path,
    log: Callable[[str], None],
) -> bool:
    if spec.distribution_name not in COPYABLE_RUNTIME_DISTRIBUTIONS:
        return False

    try:
        current_version = importlib_metadata.version(spec.distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return False
    if current_version != spec.expected_version:
        return False

    purge_module_prefixes([spec.import_name.split(".", 1)[0]])
    importlib.invalidate_caches()
    runtime_site_str = str(runtime_site.resolve())
    original_sys_path = list(sys.path)
    try:
        sys.path = [entry for entry in sys.path if entry != runtime_site_str]
        module = importlib.import_module(spec.import_name)
    except Exception:
        sys.path = original_sys_path
        return False
    finally:
        sys.path = original_sys_path

    origin = module_origin(module)
    if origin is None:
        return False
    source_entry = Path(origin)
    if source_entry.name == "__init__.py":
        source_entry = source_entry.parent

    target_entry = runtime_site / source_entry.name
    copy_path_entry(source_entry, target_entry)

    try:
        distribution = importlib_metadata.distribution(spec.distribution_name)
        distribution_path = getattr(distribution, "_path", None)
        if distribution_path:
            copy_path_entry(Path(distribution_path), runtime_site / Path(distribution_path).name)
    except importlib_metadata.PackageNotFoundError:
        pass

    log(
        f"[runtime] copied {spec.distribution_name}=={spec.expected_version} from current env into {runtime_site}"
    )
    purge_module_prefixes([spec.import_name.split(".", 1)[0]])
    importlib.invalidate_caches()
    return True


def run_pip_install(
    *,
    python_executable: str,
    target_dir: Path,
    requirements: Sequence[str],
    log: Callable[[str], None],
    index_url: Optional[str] = None,
    find_links: Optional[str] = None,
    no_deps: bool = False,
) -> None:
    command = [
        python_executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        str(target_dir),
    ]
    if no_deps:
        command.append("--no-deps")
    if index_url:
        command.extend(["--index-url", index_url])
    if find_links:
        command.extend(["--find-links", find_links])
    command.extend(requirements)

    log(f"[runtime] pip_install={' '.join(requirements)}")
    subprocess.run(command, check=True)


def install_runtime_group_with_fallback(
    *,
    python_executable: str,
    runtime_site: Path,
    specs: Sequence[RuntimePackageSpec],
    log: Callable[[str], None],
) -> None:
    if not specs:
        return

    group_index_url = specs[0].index_url
    group_find_links = specs[0].find_links
    group_no_deps = specs[0].no_deps
    requirements = [spec.install_requirement for spec in specs]

    try:
        run_pip_install(
            python_executable=python_executable,
            target_dir=runtime_site,
            requirements=requirements,
            log=log,
            index_url=group_index_url,
            find_links=group_find_links,
            no_deps=group_no_deps,
        )
        return
    except subprocess.CalledProcessError as exc:
        log(
            "[runtime] grouped install failed; retrying package-by-package "
            f"for {','.join(spec.distribution_name for spec in specs)}: {exc}"
        )

    for spec in specs:
        try:
            run_pip_install(
                python_executable=python_executable,
                target_dir=runtime_site,
                requirements=[spec.install_requirement],
                log=log,
                index_url=spec.index_url,
                find_links=spec.find_links,
                no_deps=spec.no_deps,
            )
        except subprocess.CalledProcessError as exc:
            if maybe_copy_distribution_from_current_env(spec, runtime_site, log):
                continue
            raise RuntimeError(
                "[runtime] unable to provision managed package "
                f"{spec.distribution_name!r} after pip install fallback: {exc}"
            ) from exc


def provision_managed_runtime(
    *,
    python_executable: str,
    runtime_root: Path,
    runtime_specs: Dict[str, RuntimePackageSpec],
    log: Callable[[str], None],
) -> Path:
    if runtime_root.exists():
        log(f"[runtime] removing invalid managed runtime root: {runtime_root}")
        shutil.rmtree(runtime_root)

    runtime_site = runtime_site_packages_dir(runtime_root)
    runtime_site.mkdir(parents=True, exist_ok=True)

    torch_family = [runtime_specs[name] for name in ("torch", "torchvision", "torchaudio") if name in runtime_specs]
    extras = [runtime_specs[name] for name in ("triton", "tensordict", "sgl-kernel") if name in runtime_specs]
    flashinfer_specs = [runtime_specs[name] for name in ("flashinfer-python",) if name in runtime_specs]

    install_runtime_group_with_fallback(
        python_executable=python_executable,
        runtime_site=runtime_site,
        specs=torch_family,
        log=log,
    )
    install_runtime_group_with_fallback(
        python_executable=python_executable,
        runtime_site=runtime_site,
        specs=extras,
        log=log,
    )
    run_pip_install(
        python_executable=python_executable,
        target_dir=runtime_site,
        requirements=MANAGED_RUNTIME_SUPPORT_REQUIREMENTS,
        log=log,
    )
    install_runtime_group_with_fallback(
        python_executable=python_executable,
        runtime_site=runtime_site,
        specs=flashinfer_specs,
        log=log,
    )

    return runtime_site


def probe_import_targets(
    *,
    python_executable: str,
    runtime_site: Path,
    job_overlay_dir: Path,
    import_targets: Sequence[str],
    reusable_overlay_dirs: Sequence[Path] = (),
    local_import_paths: Sequence[Path] = LOCAL_IMPORT_PATHS,
) -> ProbeResult:
    for target in import_targets:
        helper_result = run_isolated_helper(
            python_executable=python_executable,
            runtime_site=runtime_site,
            job_overlay_dir=job_overlay_dir,
            reusable_overlay_dirs=reusable_overlay_dirs,
            local_import_paths=local_import_paths,
            payload={
                "kind": "probe_targets",
                "import_targets": [target],
            },
        )
        if not helper_result.get("success"):
            return ProbeResult(
                success=False,
                target=str(helper_result.get("target") or target),
                missing_module=helper_result.get("missing_module"),
                traceback_text=str(helper_result.get("traceback_text", "")),
            )

    return ProbeResult(
        success=True,
        target=None,
        missing_module=None,
        traceback_text="",
    )


def write_constraint_file(constraint_file: Path, constraints: Sequence[str]) -> None:
    constraint_file.parent.mkdir(parents=True, exist_ok=True)
    constraint_file.write_text("\n".join(constraints) + "\n", encoding="utf-8")


def overlay_entry_requirement_key(entry_name: str) -> str:
    candidate = entry_name
    for suffix in (".dist-info", ".data", ".egg-info"):
        if candidate.endswith(suffix):
            candidate = candidate[: -len(suffix)]
            match = DIST_INFO_NAME_RE.match(candidate)
            if match:
                candidate = match.group("name")
            break
    return normalize_requirement_key(candidate)


def should_cache_overlay_entry(entry_name: str) -> bool:
    if entry_name == "__pycache__":
        return False

    requirement_key = overlay_entry_requirement_key(entry_name)
    if requirement_key in BLOCKED_REQUIREMENT_KEYS:
        return False
    if requirement_key in LOCAL_IMPORT_MODULES:
        return False
    if any(
        requirement_key == prefix or requirement_key.startswith(f"{prefix}_")
        for prefix in BLOCKED_OVERLAY_PREFIX_KEYS
    ):
        return False
    return True


def purge_blocked_overlay_entries(
    overlay_dir: Path,
    log: Callable[[str], None],
    *,
    context_label: str,
) -> int:
    if not overlay_dir.exists():
        return 0

    removed_count = 0
    for entry in overlay_dir.iterdir():
        if should_cache_overlay_entry(entry.name):
            continue
        removed_count += 1
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        log(f"{context_label} removed unsafe overlay entry {entry.name}")
    return removed_count


def sync_overlay_entries(
    source_dir: Path,
    target_dir: Path,
    log: Callable[[str], None],
    *,
    context_label: str,
    max_copy_bytes: int = OVERLAY_CACHE_MAX_COPY_BYTES,
) -> int:
    if not source_dir.exists():
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    copied_count = 0
    for entry in sorted(source_dir.iterdir(), key=lambda path: path.name):
        if not should_cache_overlay_entry(entry.name):
            continue
        entry_size = overlay_entry_size_bytes(entry)
        if entry_size > max_copy_bytes:
            log(
                f"{context_label} skipped large overlay entry {entry.name} "
                f"size_bytes={entry_size} max_copy_bytes={max_copy_bytes}"
            )
            continue
        copy_path_entry(entry, target_dir / entry.name)
        copied_count += 1

    log(f"{context_label} synced {copied_count} safe overlay entries")
    return copied_count


def snapshot_overlay_entries(overlay_dir: Path) -> Dict[str, tuple[int, int, bool]]:
    snapshot: Dict[str, tuple[int, int, bool]] = {}
    if not overlay_dir.exists():
        return snapshot

    for entry in overlay_dir.iterdir():
        if not should_cache_overlay_entry(entry.name):
            continue
        stat_result = entry.stat(follow_symlinks=False)
        snapshot[entry.name] = (
            stat_result.st_mtime_ns,
            stat_result.st_size,
            entry.is_dir(),
        )
    return snapshot


def changed_overlay_entry_names(
    overlay_dir: Path,
    before_snapshot: Dict[str, tuple[int, int, bool]],
) -> List[str]:
    changed: List[str] = []
    if not overlay_dir.exists():
        return changed

    for entry in overlay_dir.iterdir():
        if not should_cache_overlay_entry(entry.name):
            continue
        stat_result = entry.stat(follow_symlinks=False)
        current_signature = (
            stat_result.st_mtime_ns,
            stat_result.st_size,
            entry.is_dir(),
        )
        if before_snapshot.get(entry.name) != current_signature:
            changed.append(entry.name)
    return sorted(changed)


def sync_overlay_entry_names(
    source_dir: Path,
    target_dir: Path,
    entry_names: Sequence[str],
    log: Callable[[str], None],
    *,
    context_label: str,
    max_copy_bytes: int = OVERLAY_CACHE_MAX_COPY_BYTES,
) -> int:
    if not source_dir.exists():
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    copied_count = 0
    for entry_name in entry_names:
        source_path = source_dir / entry_name
        if not source_path.exists() or not should_cache_overlay_entry(entry_name):
            continue
        entry_size = overlay_entry_size_bytes(source_path)
        if entry_size > max_copy_bytes:
            log(
                f"{context_label} skipped large overlay entry {entry_name} "
                f"size_bytes={entry_size} max_copy_bytes={max_copy_bytes}"
            )
            continue
        copy_path_entry(source_path, target_dir / entry_name)
        copied_count += 1

    log(f"{context_label} synced {copied_count} changed safe overlay entries")
    return copied_count


def install_requirement(
    *,
    python_executable: str,
    overlay_dir: Path,
    requirement: str,
    log: Callable[[str], None],
    constraint_requirements: Sequence[str] = (),
) -> None:
    requirement_key = normalize_requirement_key(requirement)
    if requirement_key in BLOCKED_REQUIREMENT_KEYS:
        raise RuntimeError(
            "[bootstrap] refusing to install heavyweight runtime requirement "
            f"{requirement!r} into the job overlay; it must come from the managed runtime or local repo"
        )

    command = [
        python_executable,
        "-m",
        "pip",
        "install",
        "--upgrade",
        "--target",
        str(overlay_dir),
    ]
    if requirement_key in OVERLAY_NO_DEPS_REQUIREMENT_KEYS:
        command.append("--no-deps")
    if constraint_requirements:
        constraint_file = overlay_dir.parent / "bootstrap-constraints.txt"
        write_constraint_file(constraint_file, constraint_requirements)
        command.extend(["-c", str(constraint_file)])
    command.append(requirement)
    log(f"[bootstrap] installing requirement into job overlay: {requirement}")
    subprocess.run(command, check=True)
    importlib.invalidate_caches()


def bootstrap_runtime(
    *,
    import_targets: Sequence[str],
    requirement_index: Dict[str, str],
    install_one: Callable[[str], None],
    probe_once: Callable[[Sequence[str]], ProbeResult],
    log: Callable[[str], None],
    max_rounds: int,
) -> List[str]:
    installed_requirements: List[str] = []
    attempted_requirements_by_module: Dict[str, List[str]] = {}
    attempted_requirements_by_failure: Dict[str, List[str]] = {}
    last_result: Optional[ProbeResult] = None
    pending_result: Optional[ProbeResult] = None

    for round_index in range(1, max_rounds + 1):
        if pending_result is not None:
            result = pending_result
            pending_result = None
        else:
            result = probe_once(import_targets)
        last_result = result
        if result.success:
            log("[bootstrap] runtime import preflight succeeded")
            return installed_requirements

        missing_module_looks_valid = bool(
            result.missing_module and MODULE_LIKE_NAME_RE.fullmatch(result.missing_module)
        )
        if missing_module_looks_valid:
            candidates = candidate_requirements_for_module(result.missing_module, requirement_index)
            attempted = attempted_requirements_by_module.setdefault(result.missing_module, [])
        else:
            failure_key = f"nonmodule::{result.target}::{result.missing_module}"
            candidates = infer_requirements_from_traceback(result.traceback_text, requirement_index)
            attempted = attempted_requirements_by_failure.setdefault(failure_key, [])
        remaining_candidates = [candidate for candidate in candidates if candidate not in attempted]
        if not remaining_candidates:
            if not missing_module_looks_valid:
                raise RuntimeError(
                    "[bootstrap] import preflight failed with a dependency/version/metadata error "
                    f"while importing {result.target!r}, and no compatible repair requirement was inferred.\n"
                    f"Candidates tried: {attempted}\n"
                    f"{result.traceback_text}"
                )
            raise RuntimeError(
                "[bootstrap] no remaining package candidates for missing module "
                f"{result.missing_module!r} while importing {result.target!r}.\n"
                f"Candidates tried: {attempted}\n"
                f"{result.traceback_text}"
            )

        last_install_error: Optional[Exception] = None
        installed_this_round = False
        for requirement in remaining_candidates:
            if missing_module_looks_valid:
                log(
                    "[bootstrap] round "
                    f"{round_index}: missing module {result.missing_module!r} while importing {result.target!r}; "
                    f"trying package {requirement!r}"
                )
            else:
                log(
                    "[bootstrap] round "
                    f"{round_index}: dependency/version conflict while importing {result.target!r}; "
                    f"trying package {requirement!r}"
                )
            attempted.append(requirement)
            try:
                install_one(requirement)
            except Exception as exc:
                last_install_error = exc
                log(
                    "[bootstrap] install failed for "
                    f"{requirement!r} while repairing {result.missing_module!r}: {exc}"
                )
                continue

            installed_requirements.append(requirement)
            installed_this_round = True
            post_install_result = probe_once(import_targets)
            last_result = post_install_result
            if post_install_result.success:
                log("[bootstrap] runtime import preflight succeeded")
                return installed_requirements
            pending_result = post_install_result
            break

        if not installed_this_round:
            if not missing_module_looks_valid:
                raise RuntimeError(
                    "[bootstrap] all inferred repair requirements failed for a dependency/version/metadata error "
                    f"while importing {result.target!r}.\n"
                    f"Candidates tried: {attempted}\n"
                    f"Last install error: {last_install_error}\n"
                    f"{result.traceback_text}"
                )
            raise RuntimeError(
                "[bootstrap] all package candidates failed for missing module "
                f"{result.missing_module!r} while importing {result.target!r}.\n"
                f"Candidates tried: {attempted}\n"
                f"Last install error: {last_install_error}\n"
                f"{result.traceback_text}"
            )

    if last_result is not None:
        raise RuntimeError(
            f"[bootstrap] exceeded max bootstrap rounds ({max_rounds}) without satisfying runtime imports.\n"
            f"Last failing target: {last_result.target!r}\n"
            f"Last missing module: {last_result.missing_module!r}\n"
            f"{last_result.traceback_text}"
        )
    raise RuntimeError(
        f"[bootstrap] exceeded max bootstrap rounds ({max_rounds}) without satisfying runtime imports"
    )


def build_manifest_payload(
    *,
    runtime_root: Path,
    runtime_site: Path,
    job_overlay_dir: Path,
    job_bin_dir: Path,
    lightweight_overlay_cache_dir: Path,
    runtime_key: str,
    python_executable: str,
    python_version: str,
    runtime_specs: Dict[str, RuntimePackageSpec],
    validation_result: RuntimeValidationResult,
    status: str,
    installed_lightweight_requirements: Sequence[str],
) -> Dict[str, object]:
    return {
        "selected_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "runtime_key": runtime_key,
        "runtime_root": str(runtime_root.resolve()),
        "runtime_site_packages": str(runtime_site.resolve()),
        "job_overlay_dir": str(job_overlay_dir.resolve()),
        "job_bin_dir": str(job_bin_dir.resolve()),
        "lightweight_overlay_cache_dir": str(lightweight_overlay_cache_dir.resolve()),
        "python_executable": python_executable,
        "python_version": python_version,
        "requirements": {name: asdict(spec) for name, spec in runtime_specs.items()},
        "managed_packages": {
            name: asdict(state) for name, state in validation_result.managed_packages.items()
        },
        "local_imports": validation_result.local_imports,
        "installed_lightweight_requirements": list(installed_lightweight_requirements),
        "sys_path_head": validation_result.sys_path_head,
    }


def write_manifest_files(
    payload: Dict[str, object],
    canonical_manifest_path: Path,
    selected_manifest_path: Optional[Path],
) -> None:
    canonical_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    canonical_manifest_path.write_text(manifest_text, encoding="utf-8")

    if selected_manifest_path is not None:
        selected_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        selected_manifest_path.write_text(manifest_text, encoding="utf-8")


def make_logger(log_file: Optional[Path]) -> Callable[[str], None]:
    def log(message: str) -> None:
        print(message, flush=True)
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as fp:
                fp.write(message + "\n")

    return log


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bootstrap the Multiplex-Testing scratch-managed runtime.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--runtime-root", required=True, help="Base directory for managed runtime environments.")
    parser.add_argument("--job-overlay-dir", required=True, help="Per-job lightweight overlay directory.")
    parser.add_argument("--job-bin-dir", default=None, help="Per-job executable wrapper directory.")
    parser.add_argument("--manifest-path", required=True, help="Output path for the selected runtime manifest.")
    parser.add_argument("--max-rounds", type=int, default=24)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--repair", action="store_true")
    parser.add_argument(
        "--import-target",
        dest="import_targets",
        action="append",
        default=None,
        help="Additional import target(s) to probe during lightweight bootstrap. Defaults are used when omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime_root_base = Path(args.runtime_root).resolve()
    job_overlay_dir = Path(args.job_overlay_dir).resolve()
    job_bin_dir = (
        Path(args.job_bin_dir).resolve()
        if args.job_bin_dir
        else job_overlay_dir.parent / "bin"
    )
    manifest_output_path = Path(args.manifest_path).resolve()
    log_file = Path(args.log_file).resolve() if args.log_file else None
    log = make_logger(log_file)

    runtime_root_base.mkdir(parents=True, exist_ok=True)
    job_overlay_dir.mkdir(parents=True, exist_ok=True)
    job_bin_dir.mkdir(parents=True, exist_ok=True)

    runtime_specs = build_managed_runtime_specs()
    python_version = ".".join(str(part) for part in sys.version_info[:3])
    runtime_key = build_runtime_key(python_version, runtime_specs, LOCAL_IMPORT_PATHS)
    selected_runtime_root = select_runtime_root(runtime_root_base, runtime_key, python_version, runtime_specs)
    selected_runtime_site = runtime_site_packages_dir(selected_runtime_root)
    canonical_manifest_path = runtime_manifest_path(selected_runtime_root)
    import_targets = args.import_targets or list(DEFAULT_IMPORT_TARGETS)
    requirement_index = collect_requirement_index()
    constraint_requirements = build_lightweight_constraint_requirements(requirement_index)
    lightweight_overlay_key = build_lightweight_overlay_key(
        runtime_key,
        constraint_requirements,
        import_targets,
    )
    lightweight_overlay_cache_dir = select_lightweight_overlay_dir(runtime_root_base, lightweight_overlay_key)
    legacy_overlay_path = runtime_root_base.parent / LEGACY_OVERLAY_NAME
    compatibility_pin_summary = {
        key: requirement_index[key]
        for key in ("datasets", "huggingface_hub", "regex", "tokenizers", "urllib3")
        if key in requirement_index
    }

    log(f"[runtime] runtime_root_base={runtime_root_base}")
    log(f"[runtime] selected_runtime_root={selected_runtime_root}")
    log(f"[runtime] selected_runtime_site={selected_runtime_site}")
    log(f"[runtime] selected_runtime_key={runtime_key}")
    log(f"[runtime] job_overlay_dir={job_overlay_dir}")
    log(f"[bootstrap] lightweight_overlay_cache_dir={lightweight_overlay_cache_dir}")
    log(f"[bootstrap] lightweight_overlay_key={lightweight_overlay_key}")
    log(f"[runtime] manifest_output_path={manifest_output_path}")
    log(f"[bootstrap] lightweight_constraint_count={len(constraint_requirements)}")
    log(f"[bootstrap] compatibility_pins={json.dumps(compatibility_pin_summary, sort_keys=True)}")
    if legacy_overlay_path.exists():
        log(f"[runtime] ignoring legacy overlay at {legacy_overlay_path}")

    lightweight_overlay_cache_dir.mkdir(parents=True, exist_ok=True)
    purge_blocked_overlay_entries(
        lightweight_overlay_cache_dir,
        log,
        context_label="[bootstrap-cache]",
    )
    purge_blocked_overlay_entries(
        job_overlay_dir,
        log,
        context_label="[bootstrap]",
    )
    log(
        "[bootstrap-cache] using reusable lightweight overlay directly via PYTHONPATH "
        f"from {lightweight_overlay_cache_dir}"
    )

    validation_result = validate_managed_runtime_environment(
        python_executable=args.python,
        runtime_root=selected_runtime_root,
        runtime_site=selected_runtime_site,
        job_overlay_dir=job_overlay_dir,
        reusable_overlay_dirs=[lightweight_overlay_cache_dir],
        runtime_specs=runtime_specs,
        runtime_key=runtime_key,
        python_version=python_version,
        manifest_path_value=canonical_manifest_path,
        log=log,
    )

    status = "reused"
    if not validation_result.success:
        if args.validation_only and not args.repair:
            raise RuntimeError(
                "[runtime] managed runtime validation failed in validation-only mode. "
                "Rerun with --repair to provision the scratch-managed runtime."
            )
        if not args.repair:
            raise RuntimeError(
                "[runtime] managed runtime is invalid or missing. Rerun with --repair to provision it."
            )

        log("[runtime] provisioning or repairing scratch-managed runtime")
        provision_managed_runtime(
            python_executable=args.python,
            runtime_root=selected_runtime_root,
            runtime_specs=runtime_specs,
            log=log,
        )
        validation_result = validate_managed_runtime_environment(
            python_executable=args.python,
            runtime_root=selected_runtime_root,
            runtime_site=selected_runtime_site,
            job_overlay_dir=job_overlay_dir,
            reusable_overlay_dirs=[lightweight_overlay_cache_dir],
            runtime_specs=runtime_specs,
            runtime_key=runtime_key,
            python_version=python_version,
            manifest_path_value=canonical_manifest_path,
            log=log,
            require_manifest=False,
        )
        if not validation_result.success:
            raise RuntimeError(
                "[runtime] scratch-managed runtime is still invalid after repair:\n"
                + "\n".join(validation_result.reasons)
            )
        status = "repaired"

    log(f"[runtime] sys_path_head={json.dumps(validation_result.sys_path_head)}")
    for distribution_name, state in validation_result.managed_packages.items():
        log(
            f"[runtime] package={distribution_name} version={state.version} "
            f"origin={state.origin} dist_info={state.dist_info_path}"
        )
    for module_name, origin in validation_result.local_imports.items():
        log(f"[runtime] local_import={module_name} origin={origin}")

    installed_lightweight_requirements: List[str] = []
    installed_runtime_tool_requirements: List[str] = []

    def install_and_sync_requirement(requirement: str) -> None:
        before_snapshot = snapshot_overlay_entries(job_overlay_dir)
        install_requirement(
            python_executable=args.python,
            overlay_dir=job_overlay_dir,
            requirement=requirement,
            log=log,
            constraint_requirements=constraint_requirements,
        )
        sync_overlay_entry_names(
            job_overlay_dir,
            lightweight_overlay_cache_dir,
            changed_overlay_entry_names(job_overlay_dir, before_snapshot),
            log,
            context_label="[bootstrap-cache] updated reusable lightweight overlay",
        )

    def ensure_runtime_executable(command_name: str) -> Optional[str]:
        if shutil.which(command_name):
            resolved = shutil.which(command_name)
            log(f"[runtime-tool] using existing executable {command_name} at {resolved}")
            return resolved

        spec = RUNTIME_EXECUTABLE_REQUIREMENTS[command_name]
        helper_result = run_isolated_helper(
            python_executable=args.python,
            runtime_site=selected_runtime_site,
            job_overlay_dir=job_overlay_dir,
            reusable_overlay_dirs=[lightweight_overlay_cache_dir],
            local_import_paths=LOCAL_IMPORT_PATHS,
            payload={
                "kind": "package",
                "distribution_name": spec["module_name"],
                "import_name": spec["module_name"],
            },
        )
        if not helper_result.get("success"):
            requirement = spec["requirement"]
            log(
                f"[runtime-tool] missing executable {command_name!r}; "
                f"installing supporting package {requirement!r}"
            )
            install_and_sync_requirement(requirement)
            installed_runtime_tool_requirements.append(requirement)
            helper_result = run_isolated_helper(
                python_executable=args.python,
                runtime_site=selected_runtime_site,
                job_overlay_dir=job_overlay_dir,
                reusable_overlay_dirs=[lightweight_overlay_cache_dir],
                local_import_paths=LOCAL_IMPORT_PATHS,
                payload={
                    "kind": "package",
                    "distribution_name": spec["module_name"],
                    "import_name": spec["module_name"],
                },
            )
            if not helper_result.get("success"):
                raise RuntimeError(
                    f"[runtime-tool] failed to install module backing executable {command_name!r}.\n"
                    f"{helper_result.get('traceback_text', 'unknown helper failure')}"
                )

        candidate_paths = runtime_executable_candidate_paths(
            command_name,
            job_overlay_dir,
            reusable_overlay_dirs=[lightweight_overlay_cache_dir],
            module_origin=helper_result.get("origin"),
        )
        wrapper_path = job_bin_dir / command_name
        for candidate_path in candidate_paths:
            if candidate_path.exists() and os.access(candidate_path, os.X_OK):
                if wrapper_path.exists() or wrapper_path.is_symlink():
                    wrapper_path.unlink()
                wrapper_path.symlink_to(candidate_path)
                log(
                    f"[runtime-tool] linked executable {command_name} from {candidate_path} "
                    f"to {wrapper_path}"
                )
                return str(wrapper_path)

        candidate_path_strings = [str(path) for path in candidate_paths]
        wrapper_code = f"""#!{args.python}
import os
import sys
from pathlib import Path

candidates = [Path(path) for path in {candidate_path_strings!r}]

for candidate in candidates:
    if candidate.exists() and os.access(candidate, os.X_OK):
        os.execv(str(candidate), [str(candidate), *sys.argv[1:]])

os.execv(sys.executable, [sys.executable, "-m", {spec["module_name"]!r}, *sys.argv[1:]])
"""
        wrapper_path.write_text(wrapper_code, encoding="utf-8")
        wrapper_path.chmod(0o755)
        log(
            f"[runtime-tool] prepared wrapper {command_name} at {wrapper_path} "
            f"with candidates={[str(path) for path in candidate_paths]}"
        )
        return str(wrapper_path)

    payload = build_manifest_payload(
        runtime_root=selected_runtime_root,
        runtime_site=selected_runtime_site,
        job_overlay_dir=job_overlay_dir,
        job_bin_dir=job_bin_dir,
        lightweight_overlay_cache_dir=lightweight_overlay_cache_dir,
        runtime_key=runtime_key,
        python_executable=args.python,
        python_version=python_version,
        runtime_specs=runtime_specs,
        validation_result=validation_result,
        status=status,
        installed_lightweight_requirements=installed_lightweight_requirements,
    )
    write_manifest_files(payload, canonical_manifest_path, manifest_output_path)
    log(f"[runtime] manifest_written={canonical_manifest_path}")
    log(f"[runtime] selected_manifest_written={manifest_output_path}")

    if not args.validation_only:
        installed_lightweight_requirements = bootstrap_runtime(
            import_targets=import_targets,
            requirement_index=requirement_index,
            install_one=install_and_sync_requirement,
            probe_once=lambda targets: probe_import_targets(
                python_executable=args.python,
                runtime_site=selected_runtime_site,
                job_overlay_dir=job_overlay_dir,
                reusable_overlay_dirs=[lightweight_overlay_cache_dir],
                import_targets=targets,
                local_import_paths=LOCAL_IMPORT_PATHS,
            ),
            log=log,
            max_rounds=args.max_rounds,
        )
        for executable_name in sorted(RUNTIME_EXECUTABLE_REQUIREMENTS):
            ensure_runtime_executable(executable_name)

    payload = build_manifest_payload(
        runtime_root=selected_runtime_root,
        runtime_site=selected_runtime_site,
        job_overlay_dir=job_overlay_dir,
        job_bin_dir=job_bin_dir,
        lightweight_overlay_cache_dir=lightweight_overlay_cache_dir,
        runtime_key=runtime_key,
        python_executable=args.python,
        python_version=python_version,
        runtime_specs=runtime_specs,
        validation_result=validation_result,
        status=status,
        installed_lightweight_requirements=installed_lightweight_requirements + installed_runtime_tool_requirements,
    )
    write_manifest_files(payload, canonical_manifest_path, manifest_output_path)

    if installed_lightweight_requirements:
        log(f"[bootstrap] installed_requirements={','.join(installed_lightweight_requirements)}")
    else:
        log("[bootstrap] no additional lightweight packages were needed")
    log(f"[runtime] manifest_written={canonical_manifest_path}")
    log(f"[runtime] selected_manifest_written={manifest_output_path}")


if __name__ == "__main__":
    main()
