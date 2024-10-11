import datasets

if __name__ == '__main__':
    dataset = datasets.load_dataset('cais/mmlu', 'all')
    dataset.save_to_disk('./data/datasets/mmlu')