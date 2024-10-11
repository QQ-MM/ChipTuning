import os
from functools import partial

import argparse

from transformers import AutoTokenizer
import datasets

from transformer_chips.chip_trainer import ChipTrainer
from transformer_chips.utils import build_dataloader
from .recipe import dataset_prefix, dataset_recipes

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token

    recipe = dataset_recipes[args.recipe]
    processor = partial(recipe['process_function'], tokenizer=tokenizer, include_answer=True)
    for spl in recipe['splits']:
        if (spl == 'eval'):
            eval_dataset = datasets.load_from_disk(os.path.join(dataset_prefix, recipe['name'], recipe['splits'][spl]))
            eval_dataset = eval_dataset.map(
                processor,
                batched=True,
                batch_size=args.process_batch_size,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=False,
            )
            if (args.eval_data_limit != -1):
                eval_dataset = eval_dataset.shuffle(seed=args.seed).select(range(args.eval_data_limit))
            eval_dataloader = build_dataloader(
                eval_dataset, 
                pad_token_id=tokenizer.pad_token_id, 
                batch_size=args.eval_batch_size,
            )

    trainer = ChipTrainer(
        task_yaml_path=recipe['task_yaml'],
        chip_path=args.chip_path,
        language_model=args.base_model,
        eval_dataset=eval_dataloader,
    )

    trainer.chiped_model.load_chips(args.chip_path)

    evaluation_results = trainer.evaluate()

    tasks = trainer.tasks.keys()
    for task in tasks:
        max_chip, max_acc = evaluation_results[task]['max_chip_name'], evaluation_results[task]['max_accuracy']
        print(f'[INFO] Task {task}: Best acc = {max_acc} on chip {max_chip}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=str, required=True)
    parser.add_argument('--chip_path', type=str, required=True)
    parser.add_argument('--base_model', type=str, required=True)

    parser.add_argument('--process_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--eval_data_limit', type=int, default=-1)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)