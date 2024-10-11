# Chip Selection Strategy Experiments
Different folders refer to experiments on different benchmarks.

The result of *Fixed* and *Optimal* strategies can be obtained with scripts in `./scripts/benchmark`.
This folder only evaluates the *Validation* strategy.

Script parameters:
* `recipe`: benchmark recipe.
* `chip_path`: path to trained chips.
* `base_model`: the backbone language model. Can be a path or a huggingface model identifier.
* `process_batch_size`: batch size of `datasets.map`.
* `eval_batch_size`: batch size for evaluating chips.
* `valid_data_limit`: the size of validation set.
* `seed`: the random seed.