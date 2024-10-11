from .data import *

dataset_prefix = './data/datasets'
dataset_recipes = {
    'MMLU': {
        'name': 'mmlu',
        'task_yaml': './data/yaml/nlp/MMLU.yaml',
        'splits': {
            'train': 'auxiliary_train',
            'eval': 'test',
        },
        'process_function': process_MMLU,
    },
    'race-m': {
        'name': 'race-m',
        'task_yaml': './data/yaml/nlp/race.yaml',
        'splits': {
            'train': 'train',
            'eval': 'test',
        },
        'process_function': process_race,
    },
    'race-h': {
        'name': 'race-h',
        'task_yaml': './data/yaml/nlp/race.yaml',
        'splits': {
            'train': 'train',
            'eval': 'test',
        },
        'process_function': process_race,
    },
    'boolq': {
        'name': 'boolq',
        'task_yaml': './data/yaml/nlp/boolq.yaml',
        'splits': {
            'train': 'train',
            'eval': 'validation',
        },
        'process_function': process_boolq,
    },
    'c3': {
        'name': 'c3',
        'task_yaml': './data/yaml/nlp/c3.yaml',
        'splits': {
            'train': 'train',
            'eval': 'validation',
        },
        'process_function': process_c3,
    },
}