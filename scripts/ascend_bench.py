import argparse
import json
import time


def run_bench(args):
    from nanovllm import LLM, SamplingParams

    llm = LLM(
        args.model,
        device="npu",
        enforce_eager=True,
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.batch_size,
        num_kvcache_blocks=args.num_kvcache_blocks,
    )
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    for _ in range(args.batch_size):
        llm.add_request(args.prompt, sampling)

    prefill_seconds = None
    decode_seconds = []
    total_start = time.perf_counter()
    while not llm.is_finished():
        step_start = time.perf_counter()
        _, num_tokens = llm.step()
        step_seconds = time.perf_counter() - step_start
        if num_tokens > 0 and prefill_seconds is None:
            prefill_seconds = step_seconds
        elif num_tokens < 0:
            decode_seconds.append(step_seconds / -num_tokens)
    total_seconds = time.perf_counter() - total_start

    output_tokens = args.batch_size * args.max_tokens
    return {
        "model": args.model,
        "batch_size": args.batch_size,
        "max_tokens": args.max_tokens,
        "max_model_len": args.max_model_len,
        "ttft_seconds": prefill_seconds,
        "tpot_seconds": sum(decode_seconds) / len(decode_seconds) if decode_seconds else None,
        "total_seconds": total_seconds,
        "output_tokens_per_second": output_tokens / total_seconds,
    }


def main():
    parser = argparse.ArgumentParser(description="Minimal Ascend eager benchmark for nano-vLLM")
    parser.add_argument("--model", required=True, help="Local Hugging Face model path")
    parser.add_argument("--prompt", default="Hello, nano-vLLM Ascend.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-num-batched-tokens", type=int, default=1024)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    args = parser.parse_args()

    print(json.dumps(run_bench(args), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
