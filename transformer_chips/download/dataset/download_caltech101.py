import datasets

if __name__ == '__main__':
    dataset = datasets.load_dataset("dpdl-benchmark/caltech101")
    dataset.save_to_disk('./data/datasets/caltech101')