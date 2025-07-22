import os
import requests
from sglang.utils import launch_server_cmd
from sglang.utils import wait_for_server, print_highlight

current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['MF_MODEL_CONFIG'] = os.path.join(current_dir, 'predict_qwen2_5_7b_instruct_800l_A2.yaml')

server_process, port = launch_server_cmd(
    "python3 -m sglang.launch_server \
        --model-path /home/ckpt/qwen2.5-7b-hf \
        --host 0.0.0.0 \
        --device npu \
        --load-format mindspore \
        --max-total-tokens 20000 \
        --disable-overlap-schedule \
        --attention-backend torch_native \
        --tp-size 1 \
        --dp-size 1"
)

wait_for_server(f"http://localhost:{port}")

url = f"http://localhost:{port}/generate"
data = {"text": "What is the capital of France?"}

response = requests.post(url, json=data)
print_highlight(response.json())
