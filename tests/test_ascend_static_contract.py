from pathlib import Path
import ast
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_cuda_kernel_dependencies_are_optional():
    project = tomllib.loads(read("pyproject.toml"))["project"]
    assert "flash-attn" not in project["dependencies"]
    assert "triton>=3.0.0" not in project["dependencies"]

    optional = project["optional-dependencies"]
    assert "cuda" in optional
    assert "flash-attn" in optional["cuda"]
    assert "triton>=3.0.0" in optional["cuda"]
    assert "ascend" in optional


def test_config_exposes_explicit_device_option():
    source = read("nanovllm/config.py")
    assert "device: str = " in source


def test_model_runner_uses_backend_instead_of_cuda_literals():
    source = read("nanovllm/engine/model_runner.py")
    assert ".cuda(" not in source
    assert "torch.cuda." not in source
    assert '"nccl"' not in source
    assert "get_backend" in source


def test_model_runner_registers_device_backend_before_distributed_init():
    source = read("nanovllm/engine/model_runner.py")
    assert source.index("self.backend.setup()") < source.index("dist.init_process_group")


def test_attention_does_not_require_flash_attn_or_triton_at_import_time():
    tree = ast.parse(read("nanovllm/layers/attention.py"))
    imports = [
        node
        for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_names = []
    for node in imports:
        if isinstance(node, ast.Import):
            imported_names.extend(alias.name for alias in node.names)
        else:
            imported_names.append(node.module or "")

    assert "triton" not in imported_names
    assert "triton.language" not in imported_names
    assert "flash_attn" not in imported_names
    assert "store_kvcache_torch" in read("nanovllm/layers/attention.py")


def test_ascend_bridge_docs_and_smoke_scripts_exist():
    assert (ROOT / "docs" / "ascend-bridge.md").is_file()
    assert (ROOT / "scripts" / "ascend_smoke.py").is_file()
    assert (ROOT / "scripts" / "ascend_bench.py").is_file()


def test_ascend_scripts_use_valid_sampling_defaults():
    assert "default=0.0" not in read("scripts/ascend_smoke.py")
    assert "default=0.0" not in read("scripts/ascend_bench.py")


def test_ascend_scripts_delay_nanovllm_import_until_execution():
    bench = read("scripts/ascend_bench.py")
    top_level = bench.split("def run_bench", maxsplit=1)[0]
    assert "from nanovllm import" not in top_level
