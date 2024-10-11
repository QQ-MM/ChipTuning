import argparse
import yaml
import os
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from .recipe import dataset_recipes


class FooDataLoader:
    def __init__(self, dataset, batch_size, num_layers):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_layers = num_layers

    def __iter__(self):
        for i in range(0, len(self.dataset['hidden_states']), self.batch_size):
            embedding_slice = self.dataset['hidden_states'][i:i+self.batch_size]
            label_slice = self.dataset['labels'][i:i+self.batch_size]
            labels = torch.concatenate(label_slice)

            batch = {
                'hidden_states': [],
                'labels': labels,
            }
            
            for j in range(self.num_layers):
                t = torch.concatenate([x[j] for x in embedding_slice]).squeeze(1)
                batch['hidden_states'].append(t)
            yield batch


def main(args):
    recipe = dataset_recipes[args.recipe]
    with open(recipe['task_yaml'], 'r', encoding='utf-8') as fin:
        y = yaml.safe_load(fin)
        tasks = list(y.keys())
        num_labels = len(y[tasks[0]])
    
    with open(os.path.join(args.embedding_path, 'train_embeddings.pt'), 'rb') as fin:
        train_dataset = torch.load(fin)
        train_dataloader = FooDataLoader(
            train_dataset,
            args.train_batch_size,
            args.layer_num,
        )
    with open(os.path.join(args.embedding_path, 'eval_embeddings.pt'), 'rb') as fin:
        eval_dataset = torch.load(fin)
        eval_dataloader = FooDataLoader(
            eval_dataset,
            args.eval_batch_size,
            args.layer_num,
        )

    model = nn.ModuleList()
    if (args.chip_type == 'linear'):
        for i in range(args.layer_num):
            model.append(nn.Linear(args.embedding_dim, num_labels))
    else:
        for i in range(args.layer_num):
            model.append(nn.Sequential(
                nn.Linear(args.embedding_dim, args.mlp_chip_hidden_dim),
                nn.ReLU(),
                nn.Linear(args.mlp_chip_hidden_dim, num_labels),
            ))
    model.cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    max_acc, max_epoch = 0, 0
    test_accs = []

    # Train
    for epoch in tqdm(range(args.epoch)):
        sum_loss, steps = 0.0, 0
        for step, batch in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()
            total_loss = 0.0
            for i in range(args.layer_num):
                inputs = batch['hidden_states'][i].to('cuda')
                labels = batch['labels'].to('cuda')
                logits = model[i](inputs)
                loss = F.cross_entropy(logits, labels)
                total_loss += loss

            sum_loss += total_loss.item()
            steps += 1
            total_loss.backward()
            optimizer.step()

        avg_loss = sum_loss / steps
        print(f'Epoch {epoch}: avg loss = {avg_loss}')

        # Evaluate
        total, correct = [0 for x in range(args.layer_num)], [0 for x in range(args.layer_num)]
        for step, batch in enumerate(eval_dataloader):
            total_loss = 0.0
            with torch.no_grad():
                for i in range(args.layer_num):
                    inputs = batch['hidden_states'][i].to('cuda')
                    labels = batch['labels'].to('cuda')
                    logits = model[i](inputs)
                    prediction = torch.log_softmax(logits, dim=-1)
                    prediction = torch.argmax(prediction, dim=-1)
                    t, c = labels.shape[0], torch.sum(prediction == labels).item()
                    total[i] += t
                    correct[i] += c

        test_acc = correct[-1] / total[-1]
        test_accs.append(test_acc)

        if (test_acc > max_acc):
            max_acc = test_acc
            max_epoch = epoch

    print(f'Epoch {max_epoch}: max test acc = {max_acc}')

    task = tasks[0]
    gathered_accuracy = {
        task: {},
    }
    for i in range(args.layer_num):
        chip_name = f'{task}.{args.chip_type}.{i}'
        gathered_accuracy[task][chip_name] = correct[i] / total[i]

    if not(os.path.exists(args.out_path)):
        os.makedirs(args.out_path)

    with open(os.path.join(args.out_path, f'{args.chip_type}_accuracy.json'), 'w', encoding='utf-8') as fout:
        json.dump(gathered_accuracy, fout, ensure_ascii=False, indent=2)

    plt.plot(test_accs, label="test")
    plt.savefig(os.path.join(args.out_path, f'{args.chip_type}_test_acc_trend.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=str, required=True)
    parser.add_argument('--embedding_path', type=str, required=True)
    parser.add_argument('--chip_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    # parser.add_argument('--figure_path', type=str, default=None)

    parser.add_argument('--chip_type', choices=['linear', '2xMLP'])
    parser.add_argument('--mlp_chip_hidden_dim', type=int, default=256)
    parser.add_argument('--layer_num', type=int, default=32)
    parser.add_argument('--embedding_dim', type=int, default=4096)

    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    # parser.add_argument('--logging_steps', type=int, default=200)
    # parser.add_argument('--draw_accuracy_trends', action='store_true')
    parser.add_argument('--fp16', action='store_true')

    args = parser.parse_args()
    main(args)