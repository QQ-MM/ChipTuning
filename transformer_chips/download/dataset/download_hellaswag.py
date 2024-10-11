import datasets

if __name__ == '__main__':
    dataset = datasets.load_dataset("Rowan/hellaswag")
    dataset.save_to_disk('./data/datasets/hellaswag')