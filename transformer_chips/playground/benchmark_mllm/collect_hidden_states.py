import os
from functools import partial
import argparse
from tqdm import tqdm

import torch
from transformers import AutoProcessor, AutoModelForPreTraining
import datasets

from transformer_chips.utils import build_dataloader
from .recipe import dataset_prefix, dataset_recipes


def convert_batch(model, entry, **kwargs):
    batch = {k: v.to(model.device) for k, v in entry.items() if (v is not None)}

    for k in batch:
        if (k != 'input_ids') and (k != 'labels') and (k != 'probe_pos') and ('mask' not in k):
            batch[k] = batch[k].to(model.dtype)
    for key in kwargs:
        batch[key] = kwargs[key]

    return batch


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
            train_dataset = train_dataset.map(
                process_function,
                batched=True,
                batch_size=args.process_batch_size,
                remove_columns=train_dataset.column_names,
                # load_from_cache_file=False,
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
                # load_from_cache_file=False,
            )
            eval_dataloader = build_dataloader(
                eval_dataset, 
                pad_token_id=processor.tokenizer.pad_token_id, 
                batch_size=args.eval_batch_size,
            )

    mllm_model = AutoModelForPreTraining.from_pretrained(args.base_model)
    if (args.fp16):
        mllm_model = mllm_model.half()
    mllm_model.cuda()

    if not(os.path.exists(args.out_path)):
        os.makedirs(args.out_path)

    train_hidden_data = {
        'hidden_states': [],
        'labels': [],
    }
    for batch in tqdm(train_dataloader):
        batch = convert_batch(mllm_model, batch)
        probe_pos = batch.pop('probe_pos', None)
        labels = batch.pop('labels', None)
        with torch.no_grad():
            hidden_states = mllm_model(
                **batch,
                output_hidden_states=True,
                return_dict=True,
            )['hidden_states'][1:] # discard the raw embedding
        
        picked_hidden_states = [x[:, probe_pos, :].detach().cpu() for x in hidden_states]
        if (args.fp16):
            picked_hidden_states = [x.to(torch.float32) for x in picked_hidden_states]
        train_hidden_data['hidden_states'].append(picked_hidden_states)
        train_hidden_data['labels'].append(labels.detach().cpu())

    with open(os.path.join(args.out_path, 'train_embeddings.pt'), 'wb') as fout:
        torch.save(train_hidden_data, fout)

    eval_hidden_data = {
        'hidden_states': [],
        'labels': [],
    }
    for batch in tqdm(eval_dataloader):
        batch = convert_batch(mllm_model, batch)
        probe_pos = batch.pop('probe_pos', None)
        labels = batch.pop('labels', None)
        with torch.no_grad():
            hidden_states = mllm_model(
                **batch,
                output_hidden_states=True,
                return_dict=True,
            )['hidden_states'][1:] # discard the raw embedding
            
        picked_hidden_states = [x[:, probe_pos, :].detach().cpu() for x in hidden_states]
        if (args.fp16):
            picked_hidden_states = [x.to(torch.float32) for x in picked_hidden_states]
        eval_hidden_data['hidden_states'].append(picked_hidden_states)
        eval_hidden_data['labels'].append(labels.detach().cpu())

    with open(os.path.join(args.out_path, 'eval_embeddings.pt'), 'wb') as fout:
        torch.save(eval_hidden_data, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--train_example_limit', type=int, default=20000)

    parser.add_argument('--image_seq_length', type=int, default=576)

    parser.add_argument('--process_batch_size', type=int, default=8)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--fp16', action='store_true')

    args = parser.parse_args()
    main(args)