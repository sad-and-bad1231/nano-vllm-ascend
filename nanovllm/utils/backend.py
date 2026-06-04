from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class DeviceBackend:
    name: str
    rank: int

    @property
    def device(self) -> str:
        return f"{self.name}:{self.rank}"

    @property
    def dist_backend(self) -> str:
        return "hccl" if self.name == "npu" else "nccl"

    @property
    def supports_graph(self) -> bool:
        return self.name == "cuda"

    @property
    def module(self) -> Any:
        return getattr(torch, self.name)

    def setup(self):
        if self.name == "npu":
            try:
                import torch_npu  # noqa: F401
            except ImportError as exc:
                raise RuntimeError("Ascend backend requires torch_npu to be installed") from exc
        self.module.set_device(self.rank)

    def set_default_device(self):
        torch.set_default_device(self.name)

    def synchronize(self):
        self.module.synchronize()

    def empty_cache(self):
        self.module.empty_cache()

    def reset_peak_memory_stats(self):
        reset = getattr(self.module, "reset_peak_memory_stats", None)
        if reset is not None:
            reset()

    def memory_stats(self) -> dict[str, int]:
        stats = getattr(self.module, "memory_stats", None)
        return stats() if stats is not None else {}

    def mem_get_info(self) -> tuple[int, int]:
        mem_get_info = getattr(self.module, "mem_get_info", None)
        if mem_get_info is None:
            raise RuntimeError(
                f"{self.name} backend cannot query free memory; set num_kvcache_blocks explicitly"
            )
        return mem_get_info()

    def tensor(self, data, dtype: torch.dtype):
        return torch.tensor(data, dtype=dtype, device=self.name)

    def zeros(self, *shape: int, dtype: torch.dtype):
        return torch.zeros(*shape, dtype=dtype, device=self.name)

    def empty(self, *shape: int):
        return torch.empty(*shape, device=self.name)


def get_backend(device: str, rank: int) -> DeviceBackend:
    name = device.lower()
    if name not in ("cuda", "npu"):
        raise ValueError(f"Unsupported device backend: {device}")
    return DeviceBackend(name, rank)
