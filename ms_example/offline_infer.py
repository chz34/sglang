import os
import sys
import argparse

import sglang as sgl

parser  = argparse.ArgumentParser("sglang-mindspore offline infer")

parser.add_argument("--model_path,", metavar="--model_path", dest="model_path", 
                              required=False, default="/home/ckpt/Qwen3-8B", help="the model path", type=str)

args = parser.parse_args()

current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['MF_MODEL_CONFIG'] = os.path.join(current_dir, 'predict_qwen2_5_7b_instruct_800l_A2.yaml') 

def main():
    llm = sgl.Engine(model_path=args.model_path,
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

    sampling_params = {"temperature": 0.01, "top_p": 0.9}

    outputs = llm.generate(prompts, sampling_params)
    for prompt, output in zip(prompts, outputs):
        print("===============================")
        print(f"Prompt: {prompt}\nGenerated text: {output['text']}")


if __name__ == "__main__":
    main()
