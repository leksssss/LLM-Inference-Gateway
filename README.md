# LLM Inference Gateway

A Colab notebook demonstrating a LLM serving system with three core ideas: control/data plane separation, KV-cache-aware request
routing, and continuous batching with adaptive batch sizing.

All experiments run on a vLLM engine (TinyLlama-1.1B on T4 GPU)

## Requirements

- Google Colab Pro with T4 GPU
- Python 3.12
- vLLM

## How to run

1. Open `llm_inference_gateway.ipynb` in Colab
2. Set runtime to T4 GPU
3. Run all cells in order

## What it does

**Control plane** handles routing and admission control. Requests are hashed by prompt prefix and sent to whichever worker already has that prefix warm in its KV cache. On a cache miss, the request goes to the least-loaded worker and the mapping is recorded.

**Data plane** runs the actual inference. Each worker wraps a `vLLM AsyncLLMEngine` with `enable_prefix_caching=True`. Requests submitted concurrently are batched automatically at the GPU level by vLLM's internal scheduler.

**Adaptive batcher** measures p99 latency after each round and adjusts the maximum batch size — shrinking when p99 exceeds the SLO target, growing when there is headroom. 

## Experiments

| # | Experiment |
|---|---|
| 1 | KV-cache warm vs cold latency |
| 2 | KV-aware routing / prefix co-location |
| 3 | Continuous batching — throughput vs tail latency |
| 4 | Adaptive batch sizing via p99 feedback |

## Model

Uses `TinyLlama/TinyLlama-1.1B-Chat-v1.0` by default (~2.2 GB, fits on T4's 15 GB VRAM). To use a larger model change `MODEL_ID` in Cell 3. Models up to ~7B parameters in float16 should fit on the T4.