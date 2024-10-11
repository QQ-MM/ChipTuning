import datasets

if __name__ == '__main__':
    dataset = datasets.load_dataset("ybisk/piqa", trust_remote_code=True)
    dataset.save_to_disk('./data/datasets/piqa')