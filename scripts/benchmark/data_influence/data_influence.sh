python -m transformer_chips.playground.benchmark.data_influence \
    --recipe MMLU \
    --chip_path ./model/benchmark/MMLU_llama_7b_chips \
    --out_path ./data/figures/benchmark/steps_MMLU_llama_7b \
    --base_model ./model/Llama-2-7b-hf \
    --process_batch_size 8 \
    --train_batch_size 1 \
    --eval_batch_size 4 \
    --seed 42