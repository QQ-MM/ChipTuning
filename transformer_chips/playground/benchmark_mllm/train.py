import os
from functools import partial

import argparse

from transformers import AutoProcessor, AutoModelForPreTraining
from accelerate import Accelerator
import datasets

from transformer_chips.chip_trainer import ChipTrainer
from transformer_chips.utils import draw_accuracy_trends, build_dataloader
from .recipe import dataset_prefix, dataset_recipes

def main(args):
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)

    recipe = dataset_recipes[args.recipe]
    process_function = partial(
        recipe['process_function'], 
        processor=processor, 
        image_seq_length=args.image_seq_length, 
        include_answer=True,
    )
    for spl in recipe['splits']:
        if (spl == 'train'):
            train_dataset = datasets.load_from_disk(os.path.join(dataset_prefix, recipe['name'], recipe['splits'][spl]))
            if ('train2' in recipe['splits']):
                valid_dataset = datasets.load_from_disk(os.path.join(dataset_prefix, recipe['name'], recipe['splits']['train2']))
                train_dataset = datasets.concatenate_datasets([train_dataset, valid_dataset])
            if (args.train_example_limit != -1):
                train_dataset = train_dataset.shuffle(seed=args.seed)
                train_dataset = train_dataset.take(min(args.train_example_limit, len(train_dataset)))
            train_dataset = train_dataset.map(
                process_function,
                batched=True,
                batch_size=args.process_batch_size,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=False,
            )
            train_dataloader = build_dataloader(
                train_dataset, 
                pad_token_id=processor.tokenizer.pad_token_id, 
                batch_size=args.train_batch_size,
            )
        else:
            eval_dataset = datasets.load_from_disk(os.path.join(dataset_prefix, recipe['name'], recipe['splits'][spl]))
            eval_dataset = eval_dataset.map(
                process_function,
                batched=True,
                batch_size=args.process_batch_size,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=False,
            )
            eval_dataloader = build_dataloader(
                eval_dataset, 
                pad_token_id=processor.tokenizer.pad_token_id, 
                batch_size=args.eval_batch_size,
            )

    if (args.train_parallel):
        accelerator = Accelerator()
    else:
        accelerator = None

    mllm_model = AutoModelForPreTraining.from_pretrained(args.base_model)
    trainer = ChipTrainer(
        task_yaml_path=recipe['task_yaml'],
        chip_path=args.chip_path,
        language_model=mllm_model,
        epoch=args.epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        mlp_chip_hidden_dim=args.mlp_chip_hidden_dim,
        train_dataset=train_dataloader,
        eval_dataset=eval_dataloader,
        accelerator=accelerator,
        fp16=args.fp16,
        layer_num=args.layer_num,
        embedding_dim=args.embedding_dim,
    )

    trainer.train(
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
    )

    evaluation_results = trainer.evaluate(
        out_path=args.out_path,
    )

    tasks = trainer.tasks.keys()
    for task in tasks:
        max_chip, max_acc = evaluation_results[task]['max_chip_name'], evaluation_results[task]['max_accuracy']
        print(f'[INFO] Task {task}: Best acc = {max_acc} on chip {max_chip}')

    if (args.draw_accuracy_trends):
        draw_accuracy_trends(
            data_path=args.out_path,
            out_path=args.figure_path,
            tasks=list(trainer.tasks.keys()),
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=str, required=True)
    parser.add_argument('--chip_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--figure_path', type=str, default=None)
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--train_example_limit', type=int, default=20000)

    parser.add_argument('--mlp_chip_hidden_dim', type=int, default=256)
    parser.add_argument('--layer_num', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=4096)
    parser.add_argument('--image_seq_length', type=int, default=576)

    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--process_batch_size', type=int, default=8)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--logging_steps', type=int, default=200)
    parser.add_argument('--save_steps', type=int, default=None)
    parser.add_argument('--draw_accuracy_trends', action='store_true')
    parser.add_argument('--train_parallel', action='store_true')
    parser.add_argument('--fp16', action='store_true')

    args = parser.parse_args()
    main(args)