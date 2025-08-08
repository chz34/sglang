# MindSpore Models

## Introduction

SGLang support run MindSpore framework models, this doc guide users to run mindspore models with SGLang.

## Requiemnts

MindSpore with SGLang current only support Ascend Npu device, users need first install Ascend CANN software packages.
The CANN software packages can download from the [Ascend Offical Websites](https://www.hiascend.com).
Note: For simplify, this doc use Ascend aarch64 cpu with Atlas A2 Ascend Npu device.

Currently users need download the following packages:

```shell
Ascend-cann-toolkit_8_2.RC1_linux-aarch64.run         # the cann toolkit packages
Ascend-cann-kernels-910b_8_2.RC1_linux-aarch64.run    # the kernels so for Atlas A2 device
Ascend-cann-nnal_8_2.RC1_linux-aarch64.run            # the accerlator packages
```

Run the following command to install CANN packages:

```shell
bash Ascend-cann-toolkit_8_2.RC1_linux-aarch64.run --full --install-path=/path/to/cann
bash Ascend-cann-kernels-910b_8_2.RC1_linux-aarch64.run --install
source /path/to/cann/ascend-toolkit/set_env.sh
bash Ascend-cann-nnal_8_2.RC1_linux-aarch64.run --install
```

## Installation

Users can use conda environment to run SGLang, run the following command to create a new conda environment:

```shell
export SGLANG_CONDA_ENV_NAME=mindspore-sglang-conda-py311
conda create -n ${SGLANG_CONDA_ENV_NAME} python=3.11
conda activate ${SGLANG_CONDA_ENV_NAME}
```

Export CANN environments with the following command:

```shell
source /path/to/cann/ascend-toolkit/set_env.sh
```

Install CANN wheel packages:

```shell
pip install ${ASCEND_HOME_PATH}/lib64/te-*.whl
pip install ${ASCEND_HOME_PATH}/lib64/hccl-*.whl
pip install sympy
```

Install MindSpore wheel packages:

```shell
pip install mindspore
```

Install Torch_Npu wheel packages:

```shell
pip install torch==2.7.1
pip install torch_npu==2.7.1rc1
pip install triton_ascend
pip install torchvision==0.22.1
pip install pybind11
```

Download Sglang code and install dependents:
For aarch64, the decord is not supported, user firsh need modify the python/pyproject.toml

```diff
torch_memory_saver = ["torch_memory_saver>=0.0.8"]
- decord = ["decord"]
+ decord = []
```

```shell
cd sglang
pip install -e "python[all_cpu]"
```

## Run Model

Current SGLang-MindSpore support Qwen3 dense model, this doc uses Qwen3-8B as example.

### Download Model

The Qwen3-8B model can download from Hugging Face, run the follow command to download:

```shell
git clone https://huggingface.co/Qwen/Qwen3-8B
```

after download the model, the model directory should have the follow files:

```shell
ls
|- config.json
|- generation_config.json
|- model-00001-of-00005.safetensors
|- model-00002-of-00005.safetensors
|- model-00003-of-00005.safetensors
|- model-00004-of-00005.safetensors
|- model-00005-of-00005.safetensors
|- model.safetensors.index.json
|- tokenizer_config.json
|- tokenizer.json
|- vocab.json
```

### Prepare infer script

Write the follow script as offline_infer.py:

```python
import os
import sys
import argparse

import sglang as sgl

model_path = "/path/to/model"

def main():
    llm = sgl.Engine(model_path=model_path,
                     device="npu",
                     model_impl="mindspore",
                     max_total_tokens=20000,
                     attention_backend="ascend",
                     tp_size=1,
                     dp_size=1,
                     log_level="INFO")

    prompts = [
         "Hello, my name is",
         "The president of the United States is",
         "The capital of France is",
         "The future of AI is",
    ]

    sampling_params = {"temperature": 0.00, "top_p": 1, "top_k": 1}

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


if __name__ == "__main__":
    main()
```

### Run Infer Model

Before run model, mindspore need set some environments as follow:

```shell
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python  # to avoid protobuf binary version dismatch
export USE_VLLM_CUSTOM_ALLREDUCE=true  # to avoid using the reduce from sgl_kernel
export ASCEND_RT_VISIBLE_DEVICES=4,5,6,7  # to set device
export MS_ENABLE_LCCL=off # current not support LCCL communication mode in SGLang-MindSpore
export MS_INTERNAL_DISABLE_CUSTOM_KERNEL_LIST="PagedAttention" # use optimized PageAttention instead of mindspore native PA
```

Run the infer script to infer model:

```shell
python offline_infer.py
```
