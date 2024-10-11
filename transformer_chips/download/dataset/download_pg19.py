import datasets

if __name__ == '__main__':
    dataset = datasets.load_dataset('emozilla/pg19', trust_remote_code=True)
    dataset.save_to_disk('./data/datasets/pg19')