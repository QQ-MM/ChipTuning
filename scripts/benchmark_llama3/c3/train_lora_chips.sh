python -m transformer_chips.playground.benchmark_lora_chips.train \
    --recipe c3 \
    --chip_path ./model/benchmark_llama3/c3/llama3_8b_lora_chips \
    --out_path ./data/benchmark_llama3/c3/llama3_8b_lora_chips \
    --figure_path ./data/figures/benchmark_llama3/c3/llama3_8b_lora_chips \
    --base_model ./model/Llama-3-8B \
    --lora_path ./model/benchmark_llama3/c3/llama3_8b_lora \
    --mlp_chip_hidden_dim 256 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --process_batch_size 8 \
    --train_batch_size 1 \
    --eval_batch_size 2 \
    --epoch 1 \
    --train_example_limit 20000 \
    --logging_steps 100 \
    --draw_accuracy_trends \
    --seed 42