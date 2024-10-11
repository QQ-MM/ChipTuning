python -m transformer_chips.playground.benchmark_mllm.train_detached \
    --recipe caltech101 \
    --embedding_path ./data/benchmark_mllm/caltech101/hidden_states/llava-1.5-13b-hf \
    --chip_path ./model/benchmark_mllm/caltech101_detached/llava-1.5-13b-hf_chips \
    --out_path ./data/benchmark_mllm/caltech101_detached/llava-1.5-13b-hf \
    --chip_type linear \
    --mlp_chip_hidden_dim 5120 \
    --layer_num 40 \
    --embedding_dim 5120 \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --train_batch_size 512 \
    --eval_batch_size 512 \
    --epoch 500 \
    --seed 42

python -m transformer_chips.playground.benchmark_mllm.train_detached \
    --recipe caltech101 \
    --embedding_path ./data/benchmark_mllm/caltech101/hidden_states/llava-1.5-13b-hf \
    --chip_path ./model/benchmark_mllm/caltech101_detached/llava-1.5-13b-hf_chips \
    --out_path ./data/benchmark_mllm/caltech101_detached/llava-1.5-13b-hf \
    --chip_type 2xMLP \
    --mlp_chip_hidden_dim 5120 \
    --layer_num 40 \
    --embedding_dim 5120 \
    --learning_rate 1e-3 \
    --weight_decay 0.01 \
    --train_batch_size 512 \
    --eval_batch_size 512 \
    --epoch 500 \
    --seed 42