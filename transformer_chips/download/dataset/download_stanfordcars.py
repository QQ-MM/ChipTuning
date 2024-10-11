import datasets

if __name__ == '__main__':
    dataset = datasets.load_dataset("tanganke/stanford_cars")
    dataset.save_to_disk('./data/datasets/stanford_cars')