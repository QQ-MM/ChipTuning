import datasets

if __name__ == '__main__':
    dataset = datasets.load_dataset("dpdl-benchmark/oxford_flowers102")
    dataset.save_to_disk('./data/datasets/flowers102')