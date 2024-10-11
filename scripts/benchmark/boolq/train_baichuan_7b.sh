python -m transformer_chips.playground.benchmark.train \
    --recipe boolq \
    --chip_path ./model/benchmark/boolq/baichuan_7b_chips \
    --out_path ./data/benchmark/boolq/baichuan_7b \
    --figure_path ./data/figures/benchmark/boolq/baichuan_7b \
    --base_model ./model/Baichuan2-7B \
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