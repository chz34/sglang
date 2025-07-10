import os

from PIL import Image
import requests
import sglang_mindspore
import sglang as sgl

from sglang.srt.conversation import chat_templates
from sglang.utils import async_stream_and_merge, stream_and_merge


current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['MF_MODEL_CONFIG'] = os.path.join(current_dir, 'predict_qwen2_5_32b_instruct_800l_A2.yaml') 

def main():
    llm = sgl.Engine(model_path="/home/ckpt/Qwen2.5-32B-Instruct",
                     device="cpu",
                     load_format="mindspore",
                     max_total_tokens=20000,
                     disable_overlap_schedule=True,
                     tp_size=4,
                     dp_size=1)

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
