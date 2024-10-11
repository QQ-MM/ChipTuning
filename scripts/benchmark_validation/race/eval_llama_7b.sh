python -m transformer_chips.playground.benchmark_validation.eval \
    --recipe race-m \
    --chip_path ./model/benchmark/racem/llama_7b_chips \
    --base_model ./model/Llama-2-7b-hf \
    --process_batch_size 8 \
    --eval_batch_size 2 \
    --valid_data_limit 200 \
    --seed 42

python -m transformer_chips.playground.benchmark_validation.eval \
    --recipe race-h \
    --chip_path ./model/benchmark/raceh/llama_7b_chips \
    --base_model ./model/Llama-2-7b-hf \
    --process_batch_size 8 \
    --eval_batch_size 2 \
    --valid_data_limit 200 \
    --seed 42