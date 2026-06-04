# nano-vLLM to nano-vLLM-Ascend Bridge

This note keeps the first Ascend bring-up narrow. The target is a single-card,
eager-mode baseline before any graph capture, multi-card HCCL work, Triton
Ascend kernels, or production benchmarking.

## Source Path To Understand First

1. `LLM.generate` adds prompts as `Sequence` objects.
2. `Scheduler.schedule` chooses either prefill or decode work.
3. `BlockManager` owns physical KV blocks and writes block ids into each
   sequence's `block_table`.
4. `ModelRunner.prepare_prefill` and `prepare_decode` turn scheduled sequences
   into `input_ids`, `positions`, `slot_mapping`, `block_tables`, and
   `context_lens`.
5. `Qwen3ForCausalLM` calls `Attention`, which writes K/V into cache slots and
   computes prefill or decode attention.
6. `Sampler` samples one token, then `Scheduler.postprocess` updates cached
   token counts, appends the token, or frees blocks when a sequence finishes.

## Ascend V1 Scope

- Run with `device="npu"`, `tensor_parallel_size=1`, and `enforce_eager=True`.
- Keep CUDA as the default backend for upstream compatibility.
- Use `torch_npu` only for device initialization and ordinary tensor execution.
- Use PyTorch tensor operations as the first attention and KV-cache fallback.
- Skip CUDA Graph, multi-card HCCL, Triton Ascend kernels, quantization, and
  full benchmark suites in the first pass.

## vLLM-Ascend Patterns To Borrow

- Treat the environment as version-locked: CANN, driver, firmware, `torch`,
  `torch_npu`, and model dtype must be recorded together.
- Make device selection explicit. vLLM-Ascend exposes an NPU device path; this
  repo uses `LLM(..., device="npu", enforce_eager=True)`.
- Start from correctness before custom ops. vLLM-Ascend has industrial custom
  attention and compilation paths; this repo should only adopt those after the
  eager baseline is measurable.

## CANNLAB Time Budget

Spend 12-18 hours for the first pass:

- 1-2h: `torch_npu` import, NPU availability, tensor matmul.
- 2-3h: Qwen3-0.6B load and minimal forward path.
- 3-5h: prefill/decode with fallback attention.
- 3-4h: KV cache shape, `slot_mapping`, and `block_tables`.
- 2-3h: `LLM.generate` smoke and batch=2 smoke.
- 1-2h: baseline timing record.

Stop any single unknown after 30 minutes, save the command, environment, and
error output, then reduce it to a minimal repro before spending more NPU time.

## Required Run Records

For every NPU run, record:

- chip/card model
- driver and firmware
- CANN
- Python
- `torch`
- `torch_npu`
- model name/path and dtype
- prompt count, input length, max output tokens
- command and output
- TTFT, TPOT, total latency, and NPU memory if benchmarked

