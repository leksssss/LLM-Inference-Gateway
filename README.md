# LLM Inference Gateway

A Colab notebook demonstrating a LLM serving system with three core ideas: control/data plane separation via gRPC, KV-cache-aware request routing, and continuous batching with dynamic batch sizing.

## Requirements

- Google Colab with T4 GPU
- vLLM (`pip install vllm`)

## How to run

1. Open `llm_inference_gateway.ipynb` in Colab
2. Set runtime to T4 GPU
3. Run all cells in order

## What it does

**Control plane** (gRPC port 50051) handles routing and admission control. Incoming requests are hashed by prompt prefix and sent to whichever worker already has that prefix in its KV cache. If there's no cache hit, the request goes to the least-loaded worker.

**Data plane** (gRPC port 50052) runs the actual inference. Each worker wraps a vLLM engine and sits behind a continuous batching queue. The batch size adjusts automatically — it shrinks when P99 latency exceeds the target, and grows when there's headroom.

**KV cache manager** tracks which prompt prefixes are cached on which workers. Repeated or similar prompts get routed to the same worker to avoid recomputing attention keys and values from scratch.

## Experiments

| # | What it measures |
|---|---|
| 1 | Cache hit rate and latency speedup for repeated prompts |
| 2 | P50/P99 latency and throughput across concurrency levels 1–16 |
| 3 | How batch size adapts over time relative to the P99 target |
| 4 | gRPC round-trip verification for both servers |

## Model

Defaults to `facebook/opt-125m`. To use a larger model, change the `model_name` argument in `VLLMInferenceEngine`. The T4 has 15GB VRAM and can handle models up to ~7B parameters in float16.
