import os
import json
import yaml
import argparse


def main(args):
    with open(args.dataset_info_path, 'r', encoding='utf-8') as fin:
        js = json.load(fin)
        label_names = js['features']['label']['names']
        label_dict = {
            args.dataset_name: label_names
        }

    if not(os.path.exists(args.out_path)):
        os.makedirs(args.out_path)
    
    out_file = os.path.join(args.out_path, f'{args.dataset_name}.yaml')
    with open(out_file, 'w', encoding='utf-8') as fout:
        yaml.dump(label_dict, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset_name', type=str, required=True)
    parser.add_argument('--dataset_info_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)

    args = parser.parse_args()

    main(args)