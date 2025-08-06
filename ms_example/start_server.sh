python3 -m sglang.launch_server \
    --model-path /home/ckpt/Qwen3-8B \
    --host 0.0.0.0 \
    --device npu \
    --model-impl mindspore \
    --attention-backend torch_native \
    --dist-init-addr 127.0.0.1:29500 --nnodes 2 --node-rank 0 \
    --tp-size 4 \
    --dp-size 2