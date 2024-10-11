import os
import time
import torch
from tqdm import tqdm
import json

from transformer_chips.ChipedTransformer import ChipedTransformer
from transformer_chips.utils import build_task_features, load_task_from_yaml


class ChipTrainer:
    r"""Class for training and evaluating chiped transformers on a specific dataset.

    Args:
        task_yaml_path (`str`, *optional*):
            Path to the yaml file that describes the task.
        task_dict (`Dict`, *optional*):
            Dictionary that describes the task. The format is same to the content of the file in task_yaml_path.
        language_model (`Union(str, nn.Module)`, *optional*):
            The language model to attach chips. Should be a PretrainedModel identifier (str) or a model.
        chiped_model (`ChipedTransformer`, *optional*):
            The language model with chips already attached. Overrides language_model.
        chip_path (`str`, *optional*):
            Path to store the trained chips.
        train_dataset (`torch.utils.data.Dataloader`, *optional*):
            Dataloader to train the chips.
        eval_dataset (`torch.utils.data.Dataloader`, *optional*):
            Dataloader to eval the chips.
        epoch (`int`, *optional*, defaults to 1):
            Num of epochs in training.
        learning_rate (`float`, *optional*, defaults to 1e-5):
            The learning rate in training.
        weight_decay (`float`, *optional*, defaults to 0.01):
            The weight decay for optimizers.
        mlp_chip_hidden_dim (`int`, *optional*, defaults to 256):
            The hidden dimension of 2xMLP chips.
        accelerator (`Accelerator`, *optional*):
            Accelerator for multi-GPU training.
        fp16 (`bool`, *optional*):
            Whether to use float16 models.
        bf16 (`bool`, *optional*):
            Whether to use bfloat16 models.
    """
    def __init__(
        self,
        task_yaml_path=None,
        task_dict=None,
        language_model=None,
        chiped_model=None,
        chip_path=None,
        train_dataset=None,
        eval_dataset=None,
        epoch=1,
        learning_rate=1e-5,
        weight_decay=0.01,
        mlp_chip_hidden_dim=256,
        accelerator=None,
        fp16=False,
        bf16=False,
        **kwargs,
    ):  
        assert (task_yaml_path is not None) or (task_dict is not None), 'Either task_yaml_path or task_dict should be provided.'
        if (task_yaml_path is not None):
            self.task_dict = load_task_from_yaml(task_yaml_path)
        else:
            self.task_dict = task_dict

        self.tasks = {}
        self.label_maps = {}
        for task_name in self.task_dict:
            task, label_map = build_task_features(task_name, self.task_dict[task_name])
            self.label_maps[task_name] = label_map
            self.tasks[task_name] = task

        assert (chiped_model is not None) or (language_model is not None), 'Either chiped_model or language_model should be provided.'
        if (chiped_model is not None):
            self.chiped_model = chiped_model
        else:
            if (isinstance(language_model, str)):
                self.chiped_model = ChipedTransformer.from_pretrained(
                    pretrained_model_name_or_path=language_model,
                    trust_remote_code=True,
                    fp16=fp16,
                    bf16=bf16,
                )
            else:
                self.chiped_model = ChipedTransformer(
                    language_model, 
                    layer_num=kwargs.get('layer_num', None),
                    embedding_dim=kwargs.get('embedding_dim', None),
                    fp16=fp16,
                    bf16=bf16,
                )
        self.mlp_chip_hidden_dim = mlp_chip_hidden_dim
        self.chip_path = chip_path

        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset

        self.epoch = epoch
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.accelerator = accelerator


    def convert_batch(self, entry, **kwargs):
        batch = {k: v.to(self.chiped_model.device) for k, v in entry.items() if (v is not None)}

        for k in batch:
            if (k != 'input_ids') and (k != 'labels') and (k != 'probe_pos') and ('mask' not in k):
                batch[k] = batch[k].to(self.chiped_model.model.dtype)
        for key in kwargs:
            batch[key] = kwargs[key]

        return batch


    def train(
        self, 
        train_dataset=None,
        chip_path=None,
        logging_steps=200,
        save_steps=None,
        hidden_state_function=None,
        **kwargs,
    ):
        r"""Training the chips on a specific dataset.

        Args:
            train_dataset (`torch.utils.data.Dataloader`, *optional*):
                Dataloader to train the chips. Defaults to train_dataset during initialization. 
            chip_path (`str`, *optional*):
                Path to store the trained chips. Defaults to chip_path during initialization. 
            logging_steps (`int`, *optional*, defaults to 200):
                The interval of printing logs.
            save_steps (`int`, *optional*):
                The interval of saving checkpoints. Will only save the final checkpoint if not specified.
            hidden_state_function (`function`, *optional*):
                The function to get the hidden state of backbone language model. Defaults to forward().
        """
        if (train_dataset is None):
            train_dataset = self.train_dataset
        if (chip_path is None):
            chip_path = self.chip_path

        for task_name in self.tasks:
            task, out_dim = self.tasks[task_name]
            self.chiped_model.add_task(task, 'linear', out_dim)
            self.chiped_model.add_task(task, '2xMLP', out_dim, hidden_dim=self.mlp_chip_hidden_dim)
        
        if (self.accelerator is None):
            self.chiped_model.cuda()
        else:
            self.chiped_model.to(self.accelerator.device)
            self.chiped_model.device = self.accelerator.device

        optimizer = self.chiped_model.get_optimizer(self.learning_rate, self.weight_decay)
        if (self.accelerator is not None):
            model, optimizer, train_dataset = self.accelerator.prepare(self.chiped_model, optimizer, train_dataset)
        else:
            model = self.chiped_model

        start_time = time.time()

        accumulated_loss = 0.0
        accumulated_steps = 0
        for epoch in tqdm(range(self.epoch), desc='Epoch'):
            for i, entry in tqdm(enumerate(train_dataset), desc='Step'):
                optimizer.zero_grad()
                batch = self.convert_batch(entry, **kwargs)
                loss, logits = model(
                    return_loss=True,
                    return_accuracy=False,
                    hidden_state_function=hidden_state_function,
                    **batch,
                )

                accumulated_loss += loss.item()
                accumulated_steps += 1
                if (accumulated_steps % logging_steps == 0):
                    print(f'Step {accumulated_steps} avg loss: {accumulated_loss / logging_steps}')
                    accumulated_loss = 0.0
                
                if (self.accelerator is not None):
                    self.accelerator.backward(loss)
                else:
                    loss.backward()
                optimizer.step()

                if (save_steps is not None) and (i % save_steps == 0):
                    if (self.accelerator is not None):
                        self.accelerator.wait_for_everyone()
                    self.chiped_model.save_chips(os.path.join(chip_path, f'step_{i}'), accelerator=self.accelerator)
        
        time_elapsed = time.time()-start_time
        run_time = time_elapsed

        if (self.accelerator is not None):
            self.accelerator.wait_for_everyone()
        self.chiped_model.save_chips(chip_path, accelerator=self.accelerator)
        h = int(run_time//3600)
        m = int((run_time-(h*3600))//60)
        s = run_time-(h*3600)-(m*60)
        print(f'Run time : {h}h{m}m{s}s')


    def evaluate(
        self, 
        eval_dataset=None,
        chip_path=None,
        out_path=None,
        hidden_state_function=None,
        **kwargs,
    ):
        r"""Evaluating the chips on a specific dataset.

        Args:
            eval_dataset (`torch.utils.data.Dataloader`, *optional*):
                Dataloader to evaluate the chips. Defaults to eval_dataset during initialization. 
            chip_path (`str`, *optional*):
                Path to store the trained chips. Defaults to chip_path during initialization. 
            out_path (`str`, *optional*):
                Path to store the evaluation results.
            hidden_state_function (`function`, *optional*):
                The function to get the hidden state of backbone language model. Defaults to forward().
        """
        if (eval_dataset is None):
            eval_dataset = self.eval_dataset
        if (chip_path is None):
            chip_path = self.chip_path

        self.chiped_model.cuda()

        start_time = time.time()
        correct, total = {}, {}
        for chip in self.chiped_model.chips:
            correct[chip.get_name()] = 0
            total[chip.get_name()] = 0

        with torch.no_grad():
            for i, entry in tqdm(enumerate(eval_dataset)):
                batch = self.convert_batch(entry, **kwargs)

                logits, accuracy_dict = self.chiped_model(
                    return_loss=False,
                    return_accuracy=True,
                    hidden_state_function=hidden_state_function,
                    **batch,
                )
            
                for chip_name in accuracy_dict:
                    correct[chip_name] += accuracy_dict[chip_name]['correct']
                    total[chip_name] += accuracy_dict[chip_name]['total']

        all_accuracy = {}

        for chip_name in correct:
            accuracy = correct[chip_name] / total[chip_name]
            task = chip_name.split('.')[0]
            if (task not in all_accuracy):
                all_accuracy[task] = {}
            all_accuracy[task][chip_name] = accuracy

        for task in all_accuracy:
            max_chip_name, max_accuracy = None, 0
            for chip_name in all_accuracy[task]:
                if (all_accuracy[task][chip_name] > max_accuracy):
                    max_accuracy = all_accuracy[task][chip_name]
                    max_chip_name = chip_name

            all_accuracy[task]['max_accuracy'] = max_accuracy
            all_accuracy[task]['max_chip_name'] = max_chip_name

        if (out_path is not None):
            if not(os.path.exists(out_path)):
                os.makedirs(out_path)

            with open(os.path.join(out_path, 'accuracy.json'), 'w', encoding='utf-8') as fout:
                json.dump(all_accuracy, fout, ensure_ascii=False, indent=2)

        time_elapsed = time.time()-start_time
        run_time = time_elapsed
        h = int(run_time//3600)
        m = int((run_time-(h*3600))//60)
        s = run_time-(h*3600)-(m*60)
        print(f'Run time : {h}h{m}m{s}s')
        
        return all_accuracy