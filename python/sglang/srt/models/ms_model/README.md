# MindSpore Model Backend for SGLang

This directory contains the MindSpore implementation of model backends for SGLang, providing an alternative to PyTorch-based backends for inference on Ascend NPUs and other MindSpore-supported hardware.

## Overview

The MindSpore backend enables SGLang to run large language models using MindSpore's optimized inference engine, particularly targeting Ascend NPU hardware for high-performance inference.

## Supported Models

Currently, the following models are supported:

- **Qwen3**: Full implementation with attention optimizations and tensor parallelism
- *More models coming soon...*

## Installation

### Prerequisites

**MindSpore Installation**: Install MindSpore with Ascend NPU support
   ```bash
   pip install mindspore
   ```
### Environment Setup

TODO


## Usage

### Basic Offline Inference

The MindSpore backend uses `sgl.Engine` with specific parameters for NPU inference. Here's a basic example:

```python
import os
import sglang as sgl

# Initialize the engine with MindSpore backend
llm = sgl.Engine(
    model_path="/path/to/your/model",  # Local model path
    device="npu",                      # Use NPU device
    model_impl="mindspore",            # MindSpore implementation
    max_total_tokens=20000,            # Maximum total tokens
    attention_backend="ascend",        # Attention backend
    tp_size=1,                         # Tensor parallelism size
    dp_size=1                          # Data parallelism size
)

# Generate text
prompts = [
    "Hello, my name is",
    "The capital of France is",
    "The future of AI is"
]

sampling_params = {"temperature": 0.01, "top_p": 0.9}
outputs = llm.generate(prompts, sampling_params)

for prompt, output in zip(prompts, outputs):
    print(f"Prompt: {prompt}")
    print(f"Generated: {output['text']}")
    print("---")
```

### Distributed Inference

For multi-NPU inference with tensor parallelism:

```python
import os
import sglang as sgl

# Initialize with tensor parallelism
llm = sgl.Engine(
    model_path="/path/to/your/model",
    device="npu",
    model_impl="mindspore",
    max_total_tokens=20000,
    attention_backend="ascend",
    tp_size=2,    # Use 2 NPUs for tensor parallelism
    dp_size=1     # Data parallelism size
)

# Generate with distributed setup
prompts = ["Hello, my name is", "The president of the United States is"]
sampling_params = {"temperature": 0.01, "top_p": 0.9}
outputs = llm.generate(prompts, sampling_params)
```

### Server Mode

Launch a server with MindSpore backend:

```bash
# Basic server startup
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --tp-size 1 \
    --dp-size 1
```

For distributed server with multiple nodes:

```bash
# Multi-node distributed server
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --dist-init-addr 127.0.0.1:29500 \
    --nnodes 2 \
    --node-rank 0 \
    --tp-size 4 \
    --dp-size 2
```

### Server Client Example

```python
import os
import requests
from sglang.utils import launch_server_cmd, wait_for_server, print_highlight

# Set model configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['MF_MODEL_CONFIG'] = os.path.join(current_dir, 'predict_qwen2_5_7b_instruct_800l_A2.yaml')

# Launch server
server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server \
        --model-path /path/to/your/model \
        --host 0.0.0.0 \
        --device npu \
        --model-impl mindspore \
        --max-total-tokens=20000 \
        --attention-backend ascend \
        --tp-size 1 \
        --dp-size 1",
    port=37654)

# Wait for server to be ready
wait_for_server(f"http://localhost:{port}")

# Send request
url = f"http://localhost:{port}/generate"
data = {"text": "What is the capital of France?"}
response = requests.post(url, json=data)
print_highlight(response.json())
```

## Architecture

### Core Components

1. **MindSporeForCausalLM** (`mindspore.py`): Main wrapper class that integrates MindSpore models with SGLang's serving infrastructure
2. **Model Implementations** (`qwen3.py`): Specific model implementations with optimized attention and MLP layers
3. **Utility Functions** (`utils.py`): Tensor conversion utilities between PyTorch and MindSpore

### Memory Management

The MindSpore backend implements efficient memory management through:

- **Paged KV Cache**: Dynamic allocation and deallocation of KV cache memory
- **Memory Pooling**: Reuse of memory buffers to reduce allocation overhead

### Benchmarking

To benchmark the MindSpore backend:

```bash
# Run latency benchmark with specific parameters
python -m sglang.bench_one_batch \
    --model-path /path/to/your/model \
    --model-impl mindspore \
    --device npu \
    --attention-backend ascend \
    --tp-size 1 \
    --dp-size 1 \
    --batch 8 \
    --input-len 256 \
    --output-len 32

# Run throughput benchmark
python -m sglang.bench_offline_throughput \
    --model-path /path/to/your/model \
    --model-impl mindspore \
    --device npu \
    --attention-backend ascend \
    --tp-size 2 \
    --dp-size 1
```

## Development

TODO

## Troubleshooting

### Common Issues

1. **Memory Allocation Errors**:
   - Reduce `gpu_memory_utilization`
   - Check available NPU memory
   - Verify tensor parallelism configuration

2. **Performance Issues**:
   - Ensure MindSpore is compiled with optimizations
   - Check NPU driver and firmware versions
   - Verify environment variables are set correctly

3. **Model Loading Errors**:
   - Verify model weights are in correct format
   - Check model configuration compatibility
   - Ensure all dependencies are installed

### Debug Mode

Enable sglang debug logging by log-level argument.

```bash
python3 -m sglang.launch_server \
    --model-path /path/to/your/model \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend ascend \
    --log-level DEBUG
```

Enable mindspore info and debug logging by setting environments.

```bash
export GLOG_v=1  # INFO
export GLOG_v=0  # DEBUG
```

## Support

For issues specific to the MindSpore backend:

- Create an issue on the SGLang GitHub repository
- Join the [SGLang Slack community](https://slack.sglang.ai/)
- Check the [documentation](https://docs.sglang.ai/)

For MindSpore-specific issues:

- Refer to the [MindSpore documentation](https://www.mindspore.cn/)
