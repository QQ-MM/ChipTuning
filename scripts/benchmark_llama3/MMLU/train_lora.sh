python -m transformer_chips.playground.benchmark_lora.train \
    --recipe MMLU \
    --model_save_path ./model/benchmark_llama3/MMLU/llama3_8b_lora \
    --out_path ./data/benchmark_llama3/MMLU/llama3_8b_lora \
    --base_model ./model/Llama-3-8B \
    --max_seq_length 1500 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --r 16 \
    --lora_alpha 32 \
    --process_batch_size 8 \
    --train_batch_size 1 \
    --epoch 1 \
    --train_example_limit 20000 \
    --logging_steps 100 \
    --seed 42 \
    --bf16