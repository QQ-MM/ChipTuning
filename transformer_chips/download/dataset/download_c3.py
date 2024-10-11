import datasets

if __name__ == '__main__':
    dataset1 = datasets.load_dataset("dataset-org/c3", "dialog")
    dataset2 = datasets.load_dataset("dataset-org/c3", "mixed")
    dataset = datasets.DatasetDict({
        'train': datasets.concatenate_datasets([dataset1['train'], dataset2['train']]),
        'validation': datasets.concatenate_datasets([dataset1['validation'], dataset2['validation']]),
        'test': datasets.concatenate_datasets([dataset1['test'], dataset2['test']]),
    }) 
    dataset.save_to_disk('./data/datasets/c3')