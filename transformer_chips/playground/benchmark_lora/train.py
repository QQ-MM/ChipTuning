import os
from functools import partial

import argparse
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import LoraConfig, get_peft_model
import datasets

from .recipe import dataset_prefix, dataset_recipes

def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.base_model, trust_remote_code=True)
    if (args.bf16):
        model = model.bfloat16()
    elif (args.fp16):
        model = model.half()
    if (args.eval_raw_model):
        model.cuda()
    else:
        peft_config = LoraConfig(
            r=args.r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        model.cuda()

    recipe = dataset_recipes[args.recipe]
    process_function = partial(recipe['process_function'])
    for spl in recipe['splits']:
        if (spl == 'train'):
            train_dataset = datasets.load_from_disk(os.path.join(dataset_prefix, recipe['name'], recipe['splits'][spl]))
            if (args.train_example_limit != -1):
                train_dataset = train_dataset.shuffle(seed=args.seed)
                train_dataset = train_dataset.take(min(args.train_example_limit, len(train_dataset)))
            train_dataset = train_dataset.map(
                process_function,
                batched=True,
                batch_size=args.process_batch_size,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=False,
            )
        else:
            eval_dataset = datasets.load_from_disk(os.path.join(dataset_prefix, recipe['name'], recipe['splits'][spl]))
            eval_dataset = eval_dataset.map(
                process_function,
                batched=True,
                batch_size=args.process_batch_size,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=False,
            )

    if not(args.eval_raw_model):
        def formatting_prompts_func(example):
            output_texts = []
            for i in range(len(example['prompt'])):
                output_texts.append(example['prompt'][i])
            return output_texts

        response_template_with_context = "\nAnswer:"  # We added context here: "\n". This is enough for this tokenizer
        response_template_ids = tokenizer.encode(response_template_with_context, add_special_tokens=False)[2:]  # Now we have it like in the dataset texts: `[2277, 29937, 4007, 22137, 29901]`
        collator = DataCollatorForCompletionOnlyLM(response_template_ids, tokenizer=tokenizer)

        trainer = SFTTrainer(
            model,
            train_dataset=train_dataset,
            # eval_dataset=eval_dataset,
            args=TrainingArguments(
                output_dir="/tmp", 
                per_device_train_batch_size=args.train_batch_size,
                num_train_epochs=1,
            ),
            formatting_func=formatting_prompts_func,
            data_collator=collator,
            peft_config=peft_config,
            max_seq_length=args.max_seq_length,
            compute_metrics=None,
        )

        trainer.train()
        model.save_pretrained(save_directory=args.model_save_path)

    correct, total = 0, 0
    for example in tqdm(eval_dataset, desc='Eval'):
        query, label = example['query'], example['label']
        input_ids = tokenizer(query, return_tensors="pt").input_ids
        input_ids = input_ids.to(model.device)
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, do_sample=False, max_new_tokens=5, pad_token_id=tokenizer.eos_token_id)
            try:
                result = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                result = result.split('Answer:')[-1].split()[0].split(',')[0].split('.')[0].strip()
                if (args.recipe == 'boolq'):
                    if (label == 'yes'):
                        if (result.lower() in ['yes', 'true', '1']):
                            correct += 1
                    else:
                        if (result.lower() in ['no', 'false', '0']):
                            correct += 1
                else:
                    if (result == label):
                        correct += 1
            except:
                pass

            total += 1

    if not(os.path.exists(args.out_path)):
        os.makedirs(args.out_path)

    with open(os.path.join(args.out_path, 'lora_result.txt'), 'w', encoding='utf-8') as fout:
        accuracy = correct / total
        print(f'Accuracy: {correct} / {total} = {accuracy}')
        fout.write(f'Accuracy: {correct} / {total} = {accuracy}')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=str, required=True)
    parser.add_argument('--model_save_path', type=str)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--base_model', type=str, required=True)
    parser.add_argument('--train_example_limit', type=int, default=20000)

    parser.add_argument('--max_seq_length', type=int, default=1024)
    parser.add_argument('--learning_rate', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--process_batch_size', type=int, default=8)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--logging_steps', type=int, default=200)
    parser.add_argument('--save_steps', type=int, default=None)
    parser.add_argument('--draw_accuracy_trends', action='store_true')
    parser.add_argument('--train_parallel', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--eval_raw_model', action='store_true')

    args = parser.parse_args()
    main(args)