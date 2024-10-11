import datasets

if __name__ == '__main__':
    dataset = datasets.load_dataset("google/boolq")
    dataset.save_to_disk('./data/datasets/boolq')