python -m transformer_chips.playground.benchmark.eval \
    --recipe MMLU \
    --chip_path ./model/benchmark/MMLU/llama_7b_chips \
    --base_model ./model/Llama-2-7b-hf \
    --process_batch_size 8 \
    --eval_batch_size 4 \
    --seed 42