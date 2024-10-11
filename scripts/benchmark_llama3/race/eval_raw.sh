python -m transformer_chips.playground.benchmark_lora.train \
    --recipe race-m \
    --out_path ./data/benchmark_llama3/racem/llama3_8b_raw \
    --base_model ./model/Llama-3-8B \
    --process_batch_size 8 \
    --seed 42 \
    --eval_raw_model

python -m transformer_chips.playground.benchmark_lora.train \
    --recipe race-h \
    --out_path ./data/benchmark_llama3/raceh/llama3_8b_raw \
    --base_model ./model/Llama-3-8B \
    --process_batch_size 8 \
    --seed 42 \
    --eval_raw_model