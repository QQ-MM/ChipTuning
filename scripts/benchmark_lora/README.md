# LoRA Experiments
Different folders refer to experiments on different benchmarks.

Script parameters:
    * `recipe`: benchmark recipe.
    * `model_save_path`: path to save LoRA.
    * `out_path`: path to save evaluation results.
    * `base_model`: the backbone language model. Can be a path or a huggingface model identifier.
    * `max_seq_length`: maximum input sequence length. Tokens exceeding the limit will be truncated.
    * `learning_rate`, `weight_decay`: learning rate and weight decay of AdamW optimizers.
    * `r`, `lora_alpha`: parameters for LoRA.
    * `process_batch_size`: batch size of `datasets.map`.
    * `train_batch_size`: batch size for training chips. Recommended to 1.
    * `eval_batch_size`: batch size for evaluating chips.
    * `epoch`: training epochs.
    * `train_example_limit`: the maximum number of training data used. `-1` for no constraints.
    * `logging_steps`: logging interval.
    * `seed`: the random seed.