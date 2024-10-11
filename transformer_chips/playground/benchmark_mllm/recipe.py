from .data import *

dataset_prefix = './data/datasets'
dataset_recipes = {
    'flowers102': {
        'name': 'flowers102',
        'task_yaml': './data/yaml/mm/flowers102.yaml',
        'splits': {
            'train': 'train',
            'train2': 'validation',
            'eval': 'test',
        },
        'process_function': process_flowers_multichoice,
    },
    'caltech101': {
        'name': 'caltech101',
        'task_yaml': './data/yaml/mm/caltech101.yaml',
        'splits': {
            'train': 'train',
            'eval': 'test',
        },
        'process_function': process_caltech_multichoice,
    },
    'stanford_cars': {
        'name': 'stanford_cars',
        'task_yaml': './data/yaml/mm/stanford_cars.yaml',
        'splits': {
            'train': 'train',
            'eval': 'test',
        },
        'process_function': process_stanfordcars_multichoice,
    },
}