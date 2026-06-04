import argparse
import json
import platform
import time


def collect_env(torch, torch_npu):
    npu = torch.npu
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", "unknown"),
        "npu_available": bool(npu.is_available()),
        "npu_device_count": int(npu.device_count()),
        "npu_device_name": npu.get_device_name(0) if npu.is_available() else None,
    }


def run_tensor_smoke(torch):
    a = torch.randn(32, 32, device="npu")
    b = torch.randn(32, 32, device="npu")
    torch.npu.synchronize()
    start = time.perf_counter()
    c = a @ b
    torch.npu.synchronize()
    return {
        "matmul_shape": list(c.shape),
        "matmul_dtype": str(c.dtype),
        "matmul_seconds": time.perf_counter() - start,
    }


def run_generation_smoke(args):
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        args.model,
        device="npu",
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        num_kvcache_blocks=args.num_kvcache_blocks,
    )
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    start = time.perf_counter()
    outputs = llm.generate(args.prompt, sampling, use_tqdm=False)
    return {
        "generation_seconds": time.perf_counter() - start,
        "outputs": outputs,
    }


def main():
    parser = argparse.ArgumentParser(description="Ascend NPU smoke test for nano-vLLM")
    parser.add_argument("--model", help="Local Hugging Face model path")
    parser.add_argument("--prompt", action="append", default=["Hello, nano-vLLM Ascend."])
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--max-num-seqs", type=int, default=1)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    args = parser.parse_args()

    import torch
    import torch_npu

    result = {
        "env": collect_env(torch, torch_npu),
        "tensor_smoke": run_tensor_smoke(torch),
    }
    if args.model:
        result["generation_smoke"] = run_generation_smoke(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
