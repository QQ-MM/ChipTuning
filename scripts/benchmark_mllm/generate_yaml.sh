python ./transformer_chips/playground/benchmark_mllm/generate_yaml.py \
    --dataset_name flowers102 \
    --dataset_info_path ./data/benchmark/flowers102/train/dataset_info.json \
    --out_path ./data/yaml/mm

python ./transformer_chips/playground/benchmark_mllm/generate_yaml.py \
    --dataset_name caltech101 \
    --dataset_info_path ./data/benchmark/caltech101/train/dataset_info.json \
    --out_path ./data/yaml/mm

python ./transformer_chips/playground/benchmark_mllm/generate_yaml.py \
    --dataset_name stanford_cars \
    --dataset_info_path ./data/benchmark/stanford_cars/train/dataset_info.json \
    --out_path ./data/yaml/mm