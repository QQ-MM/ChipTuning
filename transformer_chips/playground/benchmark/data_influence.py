import os
from functools import partial

import argparse
import json

from transformers import AutoTokenizer
import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
            eval_dataloader = build_dataloader(
                eval_dataset, 
                pad_token_id=tokenizer.eos_token_id, 
                batch_size=args.eval_batch_size,
            )

    trainer = ChipTrainer(
        task_yaml_path=recipe['task_yaml'],
        language_model=args.base_model,
        eval_dataset=eval_dataloader,
    )

    step_path_list = []
    for f in os.listdir(args.chip_path):
        full_path = os.path.join(args.chip_path, f)
        if (os.path.isdir(full_path)):
            step_path_list.append(full_path)
    
    step_accuracy = []
    for step_path in step_path_list:
        step = int(step_path.split('_')[-1])*args.train_batch_size
        trainer.chiped_model.purge_chips()
        trainer.chiped_model.load_chips(step_path)
        evaluation_results = trainer.evaluate(
            out_path=args.out_path,
        )

        step_res_path = os.path.join(args.out_path, f'step_{step}_accuracy.json')
        with open(step_res_path, 'w', encoding='utf-8') as fout:
            json.dump(evaluation_results, fout)

        task = list(trainer.tasks.keys())[0]
        best_acc = evaluation_results[task]['max_accuracy']
        best_chip = evaluation_results[task]['max_chip_name']
        print(f'Step {step}: max accuracy {best_acc} at chip {best_chip}')
        step_accuracy.append({
            'step': step,
            'accuracy': best_acc,
        })

    df = pd.DataFrame.from_dict(step_accuracy)
    plot = sns.lineplot(df, x='step', y='accuracy')

    if not(os.path.exists(args.out_path)):
        os.makedirs(args.out_path)

    fig = plot.get_figure()
    fig.savefig(os.path.join(args.out_path, f'step_accuracy.png'))
    plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=str, required=True)
    parser.add_argument('--chip_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--base_model', type=str, required=True)

    parser.add_argument('--process_batch_size', type=int, default=8)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    main(args)