# Main Experiments
Different folders refer to experiments on different benchmarks.

Script parameters:
    * `recipe`: benchmark recipe.
    * `chip_path`: path to save trained chips.
    * `out_path`: path to save evaluation results.
    * `figure_path`: path to save figures.
    * `base_model`: the backbone language model. Can be a path or a huggingface model identifier.
    * `mlp_chip_hidden_dim`: the hidden dimension of 2-layer MLP chips.
    * `learning_rate`, `weight_decay`: learning rate and weight decay of AdamW optimizers.
    * `process_batch_size`: batch size of `datasets.map`.
    * `train_batch_size`: batch size for training chips. Recommended to 1.
    * `eval_batch_size`: batch size for evaluating chips.
    * `epoch`: training epochs.
    * `train_example_limit`: the maximum number of training data used. `-1` for no constraints.
    * `logging_steps`: logging interval.
    * `draw_accuracy_trends`: whether to draw accuracy trend figures for chips attached to different layers.
    * `seed`: the random seed.

## Impact of Training Dataset Scale
To conduct the experiment in Section 5.3, execute the scripts in the following order:
```shell
sh ./scripts/benchmark/data_influence/train_llama_7b.sh
sh ./scripts/benchmark/data_influence/data_influence.sh
```