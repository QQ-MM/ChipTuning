# Multimodal Experiments
Different folders refer to experiments on different benchmarks.

Notice: training chips on MLLMs requires a large batch size and epoch number, so we detach the gathering of hidden states from training chips.

Execute the scripts with the order of:
```shell
sh ./scripts/benchmark_mllm/detached_size/dataset_name/collect_hidden_states.sh
sh ./scripts/benchmark_mllm/detached_size/dataset_name/train.sh
sh ./scripts/benchmark_mllm/detached_size/dataset_name/draw.sh
```

Script parameters:
    * `recipe`: benchmark recipe.
    * `embedding_path`: path to saved hidden states.
    * `chip_path`: path to save trained chips.
    * `out_path`: path to save evaluation results.
    * `chip_type`: type of trained chips (linear or 2xMLP).
    * `mlp_chip_hidden_dim`: the hidden dimension of 2-layer MLP chips.
    * `layer_num`: number of decoder layers in the backbone model.
    * `embedding_dim`: hidden state dimension of the backbone model.
    * `learning_rate`, `weight_decay`: learning rate and weight decay of AdamW optimizers.
    * `process_batch_size`: batch size of `datasets.map`.
    * `train_batch_size`: batch size for training chips. Recommended to 1.
    * `eval_batch_size`: batch size for evaluating chips.
    * `epoch`: training epochs.
    * `logging_steps`: logging interval.
    * `seed`: the random seed.