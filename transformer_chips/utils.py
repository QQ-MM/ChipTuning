import os
import json
import yaml

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from transformer_chips.ChipedTransformer import ChipedTransformer


'''
Input: Label map: {label: id}
Output: Reversed label map: {id: label}
'''
def reverse_label_map(label_map):
    return {v: k for k, v in label_map.items()}


def build_task_features(task_name, labels):
    label_map = {}
    for i, label in enumerate(labels):
        label_map[label] = i

    task_tuple = (task_name, len(labels))
    return task_tuple, label_map


def load_task_from_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as fin:
        data = yaml.safe_load(fin)

    return data


def draw_task(
    df, 
    task, 
    max_info,
    out_path,
):
    df = df.loc[df.task == task]
    plot = sns.lineplot(df, x='layer', y='accuracy', hue='probe_type')

    # add maximum line
    axh = plt.axhline(y=max_info['acc'], color='r', linestyle='--')

    # simplify x labels
    x_ticks = plot.get_xticklabels()
    new_x_ticks = []
    for i in range(0, len(x_ticks), 2):
        new_x_ticks.append(x_ticks[i])
        new_x_ticks.append('')
    plot.set_xticklabels(new_x_ticks)

    if not(os.path.exists(out_path)):
        os.makedirs(out_path)

    fig = plot.get_figure()
    fig.savefig(os.path.join(out_path, f'accuracy_{task}.png'))
    plt.clf()
    max_acc, max_probe = max_info['acc'], max_info['probe']
    print(f'Max accuracy {max_acc} by chip {max_probe}.')


def draw_accuracy_trends(
    data_path,
    out_path,
    tasks,
):
    # probe_types = ['linear', '2xMLP']
    df_dict = {
        'task': [],
        'layer': [],
        'accuracy': [],
        'probe_type': [],
    }
    max_info = {}

    res_file_path = os.path.join(data_path, 'accuracy.json')
    with open(res_file_path, 'r', encoding='utf-8') as fin:
        js = json.load(fin)
    for task in js:
        task_acc = js[task]
        max_info[task] = {
            'acc': None,
            'probe': None,
        }
        for key in task_acc:
            if (key == 'max_accuracy'):
                max_info[task]['acc'] = task_acc[key]
            elif (key == 'max_chip_name'):
                max_info[task]['probe'] = task_acc[key]
            else:
                t, ptype, layer = key.split('.')
                df_dict['task'].append(t)
                df_dict['layer'].append(int(layer))
                df_dict['accuracy'].append(task_acc[key])
                df_dict['probe_type'].append(ptype)

    df = pd.DataFrame.from_dict(df_dict)
    for task in tasks:
        draw_task(df, task, max_info[task], out_path)


class ChipDatasetCollator:
    def __init__(self, pad_token_id, pad_side='right'):
        self.pad_token_id = pad_token_id
        assert pad_side in ('right', 'left')
        self.pad_side = pad_side

    def __call__(self, batch):
        input_ids = self.padcat_sequences([inputs.pop('input_ids') for inputs in batch],
                                     value=self.pad_token_id, pad_side=self.pad_side)
        position_ids = self.padcat_sequences([inputs.pop('position_ids', None) for inputs in batch],
                                        value=0, pad_side=self.pad_side)
        inputs_batch = {'input_ids': input_ids,
                        'position_ids': position_ids}

        if 'labels' in batch[0]:
            inputs_batch['labels'] = torch.tensor([inputs.pop('labels', None) for inputs in batch])
        if 'probe_pos' in batch[0]:
            inputs_batch['probe_pos'] = torch.tensor([inputs.pop('probe_pos', None) for inputs in batch], dtype=torch.int)

        # MLLM
        if 'pixel_values' in batch[0]:
            inputs_batch['pixel_values'] = torch.tensor([inputs.pop('pixel_values', None) for inputs in batch])

        inputs_batch['attention_mask'] = (
            self.padcat_sequences([inputs.pop('attention_mask', None) for inputs in batch], value=0, pad_side=self.pad_side))

        if 'record' in batch[0]:
            inputs_batch['record'] = [inputs.pop('record') for inputs in batch]

        others = self.collate_items(batch)
        inputs_batch.update(others)

        return inputs_batch
    

    def collate_items(self, batch):
        if not isinstance(batch, list):
            return batch

        item_type = type(batch[0])
        if not all(type(it) == item_type for it in batch[1:]):
            return batch
    
        if item_type == torch.Tensor:
            return torch.stack(batch)
        elif item_type == list:
            if all(not len(it) for it in batch):
                return None
            if all(not len(it) or all(isinstance(x, torch.Tensor) for x in it) for it in batch):
                batch = torch.stack(sum(batch, []))
            return batch
        elif item_type == tuple:
            return tuple(self.collate_items(list(b)) for b in zip(*batch))
        elif item_type == dict:
            keys = set.intersection(*(set(it.keys()) for it in batch))
            assert keys == set.union(*(set(it.keys()) for it in batch)),\
                f"collate dict items keys do not match: {keys} vs {set.union(*(set(it.keys()) for it in batch))}"
            return {k: self.collate_items([it[k] for it in batch]) for k in keys}
        else:
            return batch
        
    def padcat_sequences(self, sequences, value=0, pad_side='right'):
        if all(s is None for s in sequences):
            return None
        if all(type(s) == list for s in sequences):
            sequences = [torch.tensor(s, dtype=torch.int) for s in sequences]
        max_l = max(s.size(0) for s in sequences)
        sequences_ = []
        for seq in sequences:
            if seq.size(0) != max_l:
                pad_len = max_l - seq.size(0)
                pad_len = (0, pad_len) if pad_side == 'right' else (pad_len, 0)
                seq = F.pad(seq, pad_len, value=value)
            sequences_.append(seq)

        sequences = torch.stack(sequences_)

        return sequences
    

def build_dataloader(
    dataset,
    pad_token_id,
    batch_size,
    **kwargs,
):
    collator = ChipDatasetCollator(
        pad_token_id,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        **kwargs,
    )

    return dataloader


def build_pruned_chip_model(
    base_model,
    chip_path,
    chip_data_path=None,
    chip_list=None,
    layer_num=None,
    embedding_dim=None,
):
    if (chip_list is None): # load optimal chip automatically
        chip_list = []
        accuracy_path = os.path.join(chip_data_path, 'accuracy.json')
        with open(accuracy_path, 'r', encoding='utf-8') as fin:
            js = json.load(fin)
            for task_name in js:
                chip_name = js[task_name]['max_chip_name']
                chip_list.append(chip_name)

    chiped_model = ChipedTransformer(
        base_model, 
        layer_num=layer_num,
        embedding_dim=embedding_dim,
    )
    chiped_model.load_chips(
        chip_path,
        chip_names=chip_list,
    )
    chiped_model.prune_model()

    return chiped_model, chip_list