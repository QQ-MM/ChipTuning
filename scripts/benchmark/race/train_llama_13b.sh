python -m transformer_chips.playground.benchmark.train \
    --recipe race-m \
    --chip_path ./model/benchmark/racem/llama_13b_chips \
    --out_path ./data/benchmark/racem/llama_13b \
    --figure_path ./data/figures/benchmark/racem/llama_13b \
    --base_model ./model/Llama-2-13b-hf \
    --mlp_chip_hidden_dim 256 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --process_batch_size 8 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --epoch 1 \
    --train_example_limit 20000 \
    --logging_steps 100 \
    --draw_accuracy_trends \
    --seed 42 \
    --fp16

python -m transformer_chips.playground.benchmark.train \
    --recipe race-h \
    --chip_path ./model/benchmark/raceh/llama_13b_chips \
    --out_path ./data/benchmark/raceh/llama_13b \
    --figure_path ./data/figures/benchmark/raceh/llama_13b \
    --base_model ./model/Llama-2-13b-hf \
    --mlp_chip_hidden_dim 256 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --process_batch_size 8 \
    --train_batch_size 1 \
    --eval_batch_size 1 \
    --epoch 1 \
    --train_example_limit 20000 \
    --logging_steps 100 \
    --draw_accuracy_trends \
    --seed 42 \
    --fp16