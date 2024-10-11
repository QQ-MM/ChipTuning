python -m transformer_chips.playground.benchmark_lora.train \
    --recipe boolq \
    --out_path ./data/benchmark_llama3/boolq/llama3_8b_raw \
    --base_model ./model/Llama-3-8B \
    --process_batch_size 8 \
    --seed 42 \
    --eval_raw_model