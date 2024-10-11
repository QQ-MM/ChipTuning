import json
import argparse
import pandas as pd

from transformer_chips.utils import draw_task

def draw_accuracy_trends(
    df_dict,
    out_path,
    tasks,
):
    df = pd.DataFrame.from_dict(df_dict)
    for task in tasks:
        draw_task(df, task, max_info, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    args = parser.parse_args()

    result_files = {
        'flowers102': [
            './data/benchmark_mllm/flowers102_detached/llava-1.5-7b-hf/linear_accuracy.json',
            './data/benchmark_mllm/flowers102_detached/llava-1.5-7b-hf/2xMLP_accuracy.json',
        ],
        'caltech101': [
            './data/benchmark_mllm/caltech101_detached/llava-1.5-7b-hf/linear_accuracy.json',
            './data/benchmark_mllm/caltech101_detached/llava-1.5-7b-hf/2xMLP_accuracy.json',
        ],
        'stanford_cars': [
            './data/benchmark_mllm/stanford_cars_detached/llava-1.5-7b-hf/linear_accuracy.json',
            './data/benchmark_mllm/stanford_cars_detached/llava-1.5-7b-hf/2xMLP_accuracy.json',
        ],
    }
    df_dict = {
        'task': [],
        'layer': [],
        'accuracy': [],
        'probe_type': [],
    }
    max_info = {
        'acc': 0,
        'probe': '',
    }
    max_acc, max_chip = 0, None
    for file_path in result_files[args.task]:
        with open(file_path, 'r', encoding='utf-8') as fin:
            js = json.load(fin)
            js = js[args.task]
            for key in js:
                task, probe, layer = key.split('.')
                df_dict['task'].append(task)
                df_dict['layer'].append(int(layer))
                df_dict['probe_type'].append(probe)
                df_dict['accuracy'].append(js[key])

                if (js[key] > max_info['acc']):
                    max_info['acc'] = js[key]
                    max_info['probe'] = key

    draw_accuracy_trends(
        df_dict,
        args.out_path,
        [args.task],
    )