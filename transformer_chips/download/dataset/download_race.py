import datasets

if __name__ == '__main__':
    dataset = datasets.load_dataset("ehovy/race", 'high')
    dataset.save_to_disk('./data/datasets/race-h')

    dataset = datasets.load_dataset("ehovy/race", 'middle')
    dataset.save_to_disk('./data/datasets/race-m')