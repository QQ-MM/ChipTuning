CUDA_VISIBLE_DEVICES=1 python -m transformer_chips.playground.benchmark_mllm.collect_hidden_states \
    --recipe stanford_cars \
    --out_path ./data/benchmark_mllm/stanford_cars/hidden_states/llava-1.5-7b-hf \
    --base_model ./model/llava-1.5-7b-hf \
    --image_seq_length 576 \
    --process_batch_size 8 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --train_example_limit -1 \
    --seed 42