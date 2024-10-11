# Chip-Tuning

Chip-Tuning is a part of QQ MLLM project and this is the repository for paper [Chip-Tuning: Classify Before Language Models Say](https://arxiv.org/abs/2410.06541) with code and scripts.

## Setup
To use and evaluate chip-tuning, you have to install the dependencies by:

```shell
pip install -e requirements.txt
```

And download related datasets and models by excecuting python code files in `./transformer_chips/download`.
An example for downloading the MMLU dataset is:
```shell
python ./transformer_chips/download/dataset/download_MMLU.py
```
This will download the MMLU dataset and save the dataset to `./data/datasets`.

You can also manually download the resources, then put datasets under `./data/datasets` and models under `./model`.

## Training & Evaluation
The scripts for experiments in the paper are stored in `./scripts`:
- `./scripts/benchmark`: the main experiment (Section 4.2);
- `./scripts/benchmark_mllm`: the multimodal model experiments (Section 4.3);
- `./scripts/benchmark_lora` & `./scripts/benchmark_lora_chips`: combination with finetuning (Section 4.4);
- `./scripts/benchmark_validation`: chip selection strategy (Section 5.2);
- `./scripts/benchmark/data_influence`: impact of training dataset scale (Section 5.3);
- `./scripts/benchmark_llama3`: Llama3 experiments (Appendix E);

See `README.md` under these folders for more details.

We also provide examples of solely evaluating trained chips on certain benchmark or making prediction with selected chips.
See `./scripts/benchmark/misc` for examples.

## How to Add New Custom Tasks
You can add your custom tasks with two steps:
* Define your classification task with the format in `./data/yaml`;
* Implement your own recipe and data process function in `recipe.py` and `data.py`.

You can also directly use the `ChipTrainer` class in `./transformer_chips/chip_trainer.py` by passing your own `task_dict`, `train_dataset` and `eval_dataset`;
or use the `ChipedTransformer` class in `./transformer_chips/ChipedTransformer.py` to create your own trainer.

## Generation Chips (Unfinished)
Different from classification chips (`ChipedTransformer`), the generation chips (`ChipedTransformerForGeneration`) currently need further exploration.
Using them could lead to unexpected results.
