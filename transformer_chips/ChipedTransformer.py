import os
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput

from .chip import Chip, GenerationChip


@dataclass
class ChipedModelOutput(ModelOutput):
    logits: Dict[str, torch.Tensor] = None
    loss: Optional[torch.FloatTensor] = None
    accuracy: Optional[Dict[str, Dict]] = None


@dataclass
class ChipedCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    z_loss: torch.FloatTensor = None
    aux_loss: torch.FloatTensor = None
    router_logits: Optional[Tuple[torch.FloatTensor]] = None


class ChipMixin:
    r"""
    A class containing all functions for chip-tuning, to be used as a mixin in [`PreTrainedModel`].

    Args:
        model (`transformers.PreTrainedModel` or LM-like objects):
            The language model to attach chips on.
        layer_num (`int`, *optional*):
            The number of layers of the language model. Defaults to model.config.num_hidden_layers.
        embedding_dim (`int`, *optional*):
            The hidden dimension of the language model. Defaults to model.config.hidden_size.
        tasks (`List`, *optional*):
            The tasks to train chips on.
        fp16 (`bool`, *optional*):
            Whether to use float16 models.
        bf16 (`bool`, *optional*):
            Whether to use bfloat16 models.

        criterion (`function`, *optional*):
            The loss function.
    """

    def __init__(
        self, 
        model: PreTrainedModel,
        layer_num: int = None,
        embedding_dim: int = None,
        tasks: List = None,
        fp16: bool = False,
        bf16: bool = False,
        **kwargs, 
    ):
        self.chips = nn.ModuleList()
        self.task_ids = {}
        self.criterion = kwargs.get('criterion', F.cross_entropy)
        self.model = model

        self.fp16 = fp16
        self.bf16 = bf16

        if (bf16) and (isinstance(self.model, PreTrainedModel)):
            self.model = self.model.bfloat16()
        elif (fp16) and (isinstance(self.model, PreTrainedModel)):
            self.model = self.model.half()

        if (embedding_dim is not None):
            self.embedding_dim = embedding_dim
        else:
            self.embedding_dim = model.config.hidden_size
        if (self.embedding_dim is None):
            raise ValueError('Embedding size of transformers not provided.')
        
        if (layer_num is not None):
            self.layer_num = layer_num
        else:
            self.layer_num = model.config.num_hidden_layers
        if (self.layer_num is None):
            raise ValueError('Number of layers of transformers not provided.')
        
        if (tasks is not None):
            for task in tasks:
                if (task not in self.task_ids):
                    self.task_ids[task] = len(self.task_ids)

    def get_task_id(self, task: str):
        return self.task_ids.get(task, None)
    
    def get_num_tasks(self):
        return len(self.task_ids)
    
    def get_chip_names(self):
        return [chip.get_name() for chip in self.chips]

    # applied_tasks: list of task [n]
    def build_task_mask(self, applied_tasks):
        task_num = self.get_num_tasks()
        task_mask = torch.zeros((task_num,), dtype=torch.bool).to(self.device)
        for task in applied_tasks:
            task_id = self.get_task_id(task)
            if (task_id is not None):
                task_mask[task_id] = True

        return task_mask

    # applied_tasks: batched list of task [bsz, n]
    def build_task_masks(self, applied_tasks):
        task_num = self.get_num_tasks()
        task_mask = torch.zeros((len(applied_tasks), task_num), dtype=torch.bool).to(self.device)
        for i, entry in enumerate(applied_tasks):
            for task in entry:
                task_id = self.get_task_id(task)
                if (task_id is not None):
                    task_mask[i, task_id] = True

        return task_mask

    def add_chip(self, chip: Chip):
        chip = chip.to(self.device)

        task = chip.config.task
        if (task is None):
            raise ValueError('Task of chip not provided.')
        
        if (task not in self.task_ids):
            self.task_ids[task] = len(self.task_ids)
        self.chips.append(chip)

        return chip

    def add_chips(self, chips: list):
        for chip in chips:
            self.add_chip(chip)

    def add_task(self, task, chip_type, out_dim=-1, hidden_dim=-1):
        raise NotImplementedError
    
    # Remove all chips
    def purge_chips(self):
        self.chips = nn.ModuleList()
        self.task_ids = {}

    def load_chips(self, chip_path, chip_names=None):
        if (chip_names is None): # load all chips
            files = os.listdir(chip_path)
            for file in files:
                if (file.endswith('.chip')):
                    full_path = os.path.join(chip_path, file)
                    with open(full_path, 'rb') as fin:
                        chip = torch.load(fin)
                        self.add_chip(chip)
        else: # load specific chips
            for chip_name in chip_names:
                full_path = os.path.join(chip_path, f'{chip_name}.chip')
                with open(full_path, 'rb') as fin:
                    chip = torch.load(fin)
                    self.add_chip(chip)

    def save_chips(self, chip_path, chip_names=None, accelerator=None):
        if not(os.path.exists(chip_path)):
            os.makedirs(chip_path)

        if (chip_names is None): # save all chips
            for chip in self.chips:
                chip_name = chip.get_name()
                full_path = os.path.join(chip_path, f'{chip_name}.chip')
                if (accelerator is None) or (accelerator.is_main_process):
                    with open(full_path, 'wb') as fout:
                        torch.save(chip, fout)
        else: # save specified chips
            for chip in self.chips:
                chip_name = chip.get_name()
                if (chip_name in chip_names):
                    full_path = os.path.join(chip_path, f'{chip_name}.chip')
                    if (accelerator is None) or (accelerator.is_main_process):
                        with open(full_path, 'wb') as fout:
                            torch.save(chip, fout)

    def prune_model(self, cut_to_layer=None):
        if not(isinstance(self.model, PreTrainedModel)):
            raise NotImplementedError('Only model pruning on transformers.PretrainedModel is supported.')

        if (cut_to_layer is not None):
            self.model.model.layers = self.model.model.layers[:cut_to_layer+1+1] # prevent final norm by keeping one more layer
            self.layer_num = cut_to_layer+1
        elif (len(self.chips) == 0):
            return
        else:
            max_layer = -1
            for chip in self.chips:
                max_layer = max(chip.config.layer, max_layer)

            if (max_layer > 0):
                self.model.model.layers = self.model.model.layers[:max_layer+1+1]
                self.layer_num = max_layer+1

    def get_optimizer(self, learning_rate=1e-5, weight_decay=0.01):
        optimizer = torch.optim.AdamW(
            self.chips.parameters(), lr=learning_rate, betas=(0.9, 0.99), weight_decay=weight_decay
        )
        return optimizer
        


class ChipedTransformer(nn.Module, ChipMixin):
    r"""
    A language model with chips attached for classification tasks.

    Args:
        model (`transformers.PreTrainedModel` or LM-like objects):
            The language model to attach chips on.
        layer_num (`int`, *optional*):
            The number of layers of the language model. Defaults to model.config.num_hidden_layers.
        embedding_dim (`int`, *optional*):
            The hidden dimension of the language model. Defaults to model.config.hidden_size.
        tasks (`List`, *optional*):
            The tasks to train chips on.
        fp16 (`bool`, *optional*):
            Whether to use float16 models.
        bf16 (`bool`, *optional*):
            Whether to use bfloat16 models.

        criterion (`function`, *optional*):
            The loss function.
    """

    def __init__(
        self, 
        model: PreTrainedModel,
        layer_num: int = None,
        embedding_dim: int = None,
        tasks: List = None,
        fp16: bool = False,
        bf16: bool = False,
        **kwargs,
    ):  
        nn.Module.__init__(self)
        ChipMixin.__init__(
            self,
            model=model,
            layer_num=layer_num,
            embedding_dim=embedding_dim,
            tasks=tasks,
            fp16=fp16,
            bf16=bf16,
            **kwargs,
        )
        
        if (isinstance(self.model, nn.Module)):
            self.model.requires_grad_(False)

        if (isinstance(self.model, PreTrainedModel)):
            self.device = self.model.device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def cuda(self):
        super().cuda()
        if (isinstance(self.model, PreTrainedModel)):
            self.device = self.model.device
        else:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def add_task(self, task, chip_type, out_dim=-1, hidden_dim=-1):
        if (chip_type == '2xMLP') and (hidden_dim == -1):
            raise ValueError('Hidden dimension of MLP probes not provided.')
        
        for layer in range(self.layer_num):
            config_dict = {
                'task': task,
                'chip_type': chip_type,
                'in_dim': self.embedding_dim,
                'out_dim': out_dim,
                'hidden_dim': hidden_dim,
                'layer': layer,
            }
            self.add_chip(Chip.from_dict(config_dict))

    def forward(
        self, 
        task_mask=None,
        labels=None,
        probe_pos=None,
        return_loss=True,
        return_accuracy=False,
        return_dict=False,
        hidden_state_function=None,
        **kwargs,
    ):
        r"""Running a chiped model.

        Args:
            task_mask (`torch.tensor`, *optional*):
                A tensor of shape (bsz, num_tasks) that marks which tasks each data entry belongs to. Defaults to torch.ones((bsz, 1)).
            labels (`torch.tensor`, *optional*):
                The class of data entries. Required when outputting loss or accuracy. 
            probe_pos (`torch.tensor`, *optional*):
                The token position to attach probes. Defaults to all -1 (the last token).
            return_loss (`bool`, *optional*):
                Whether to calculate and return loss. Defaults to True.
            return_accuracy (`bool`, *optional*):
                Whether to calculate and return accuracy. Defaults to False.
            return_dict (`bool`, *optional*):
                Whether to return the outputs in a dict. Defaults to False.
            hidden_state_function (`function`, *optional*):
                The function to get the hidden state of backbone language model. Defaults to calling self.model.forward() and discarding the raw embedding layer.
        """

        if (hidden_state_function is None):
            _ = kwargs.pop('output_hidden_states', True)
            hidden_states = self.model(
                output_hidden_states=True,
                return_dict=True,
                **kwargs,
            )['hidden_states'][1:] # discard the raw embedding
            
            hidden_states = [e.to(torch.float32) for e in hidden_states]
        else:
            hidden_states = hidden_state_function(
                **kwargs,
            )

        all_logits = {}
        total_loss = 0.0 if return_loss else None
        accuracy_dict = {} if return_accuracy else None
        for chip in self.chips:
            layer = chip.config.layer

            hidden = hidden_states[layer].detach()
            logits = chip(hidden, probe_pos)

            all_logits[chip.get_name()] = logits

            if (labels is not None):
                if (len(labels.shape) == 1):
                    labels = labels.unsqueeze(-1)

                task_id = self.task_ids[chip.config.task]
                if (task_mask is None):
                    task_mask = torch.ones(labels.shape, dtype=torch.bool, device=self.device)
                label_mask = task_mask[:, task_id]

                if (return_loss):
                    chip_labels = labels[:, task_id] * label_mask
                    chip_logits = logits * label_mask.unsqueeze(-1)
                    if (chip_labels.shape[0] != 0):
                        chip_loss = self.criterion(chip_logits, chip_labels, reduction='none')
                        chip_loss = torch.mean(chip_loss * label_mask)
                        total_loss += chip_loss

                if (return_accuracy):
                    predictions = F.log_softmax(logits, dim=-1)
                    correct = torch.sum((torch.argmax(predictions, dim=-1) == labels[:, task_id]) * label_mask).item()
                    total = torch.sum(label_mask).item()
                    accuracy_dict[chip.get_name()] = {
                        'correct': correct,
                        'total': total,
                        'accuracy': 0 if (total == 0) else (correct / total),
                    }

        if not return_dict:
            return tuple(v for v in [total_loss, all_logits, accuracy_dict] if v is not None)
        return ChipedModelOutput(
            logits=all_logits,
            loss=total_loss,
            accuracy=accuracy_dict,
        )
    
    def prune_model(self, cut_to_layer=None, key_to_layer=None):
        r"""Prune the model, removing layers subsequent to selected chip.
        
        Args:
            cut_to_layer (`int`, *optional*):
                Number of layers to retain. Layers subsequent to cut_to_layer will be discarded. Defaults to the layer of the optimal chip.
            key_to_layer (`str`, *optional*):
                The attribute name to decoder layers. Defaults to 'model.layers'.
        """
        if (cut_to_layer is not None):
            if (key_to_layer is None):
                # FIXME: Prevent final norm by keeping one more layer. Fix it if has better solutions.
                self.model.model.layers = self.model.model.layers[:cut_to_layer+1+1]
            else:
                exec(f'self.model.{key_to_layer} = self.model.{key_to_layer}[:cut_to_layer+1+1]')
            self.layer_num = cut_to_layer+1
        elif (len(self.chips) == 0):
            return
        else:
            max_layer = -1
            for chip in self.chips:
                max_layer = max(chip.config.layer, max_layer)

            if (max_layer > 0):
                if (key_to_layer is None):
                    self.model.model.layers = self.model.model.layers[:max_layer+1+1]
                else:
                    exec(f'self.model.{key_to_layer} = self.model.{key_to_layer}[:max_layer+1+1]')
                self.layer_num = max_layer+1


    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        chip_path = None,
        chip_names = None,
        layer_num = None,
        embedding_dim = None,
        prune_model = False,
        fp16 = False,
        bf16 = False,
        **kwargs,
    ):
        base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        chiped_model = cls(base_model, layer_num, embedding_dim, fp16=fp16, bf16=bf16)

        if (chip_path is not None):
            chiped_model.load_chips(chip_path, chip_names)

        if (prune_model):
            chiped_model.prune_model()
        return chiped_model
    

class ChipedTransformerForGeneration(PreTrainedModel, ChipMixin):
    r"""
    A language model with chips attached for generation tasks.

    *Warning: this class is experimental and could lead to unexpected outputs.*

    Args:
        model (`PreTrainedModel`):
            The language model to attach chips on.
        layer_num (`int`, *optional*):
            The number of layers of the language model. Defaults to model.config.num_hidden_layers.
        embedding_dim (`int`, *optional*):
            The hidden dimension of the language model. Defaults to model.config.hidden_size.
        tasks (`List`, *optional*):
            The tasks to train chips on.
        fp16 (`bool`, *optional*):
            Whether to use float16 models.
        bf16 (`bool`, *optional*):
            Whether to use bfloat16 models.

        criterion (`function`, *optional*):
            The loss function.
    """
    def __init__(
        self, 
        model: PreTrainedModel,
        layer_num: int = None,
        embedding_dim: int = None,
        vocab_size: int = None,
        tasks: List = None,
        **kwargs,
    ):
        config = kwargs.pop('config', None)
        if (config is None):
            config = model.config

        PreTrainedModel.__init__(
            self,
            config=config,
            **kwargs,
        )
        ChipMixin.__init__(
            self,
            model=model,
            layer_num=layer_num,
            embedding_dim=embedding_dim,
            tasks=tasks,
            **kwargs,
        )

        if (vocab_size is not None):
            self.vocab_size = vocab_size
        else:
            self.vocab_size = self.model.vocab_size

        self.criterion = kwargs.get('criterion', F.cross_entropy)
        if (isinstance(self.model, nn.Module)):
            self.model.requires_grad_(False)

    def add_task(self, task, begin_layer=0, chip_type='generation', lm_head=None, **kwargs):
        for layer in range(begin_layer, self.layer_num):
            config_dict = {
                'task': task,
                'chip_type': chip_type,
                'in_dim': self.embedding_dim,
                'out_dim': self.vocab_size,
                'hidden_dim': kwargs.get('hidden_dim', -1),
                'layer': layer,
            }
            chip = self.add_chip(GenerationChip.from_dict(config_dict))
            with torch.no_grad():
                if (lm_head is None):
                    chip.lm_head.weight.copy_(self.model.lm_head.weight)
                else:
                    chip.lm_head.weight.copy_(lm_head.weight)

    def forward(
        self,
        return_loss=True,
        return_dict=False,
        norm_function=None,
        lm_head=None,
        generation_chip=None,
        **kwargs,
    ):
        _, _ = kwargs.pop('output_hidden_states', True), kwargs.pop('return_dict', True)
        model_outputs = self.model(
            output_hidden_states=True,
            return_dict=True,
            **kwargs,
        )
        hidden_states = model_outputs['hidden_states'][1:] # discard the raw embedding layer

        if (norm_function is None):
            norm_function = self.model.model.norm
        if (lm_head is None):
            lm_head = self.model.lm_head

        output_logits = None
        total_loss = 0.0 if return_loss else None

        for chip in self.chips:
            layer = chip.config.layer
            hidden = hidden_states[layer].detach()
            if (layer+1 != self.layer_num): # apply norm on non-final layers
                logits = chip(hidden, norm_function=norm_function)
            else:
                logits = chip(hidden, norm_function=None)

            if (generation_chip == chip.get_name()):
                output_logits = logits

            if (return_loss):
                labels = kwargs.pop('labels', None)
                if (labels is None):
                    continue
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                chip_loss = self.criterion(shift_logits, shift_labels)
                chip_loss = torch.nan_to_num(chip_loss)
                total_loss += chip_loss
        
        logits = output_logits
        if not return_dict:
            output = (logits,) + model_outputs[1:]
            return (total_loss,) + output if total_loss is not None else output

        return ChipedCausalLMOutput(
            loss=total_loss,
            logits=logits,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        inputs_embeds=None,
        **kwargs,
    ):
        used_keys = ['task_mask', 'norm_function', 'lm_head', 'generation_chip']
        used_values = {}
        for key in used_keys:
            value = kwargs.pop(key, None)
            if (value is not None):
                used_values[key] = value
                
        inputs = self.model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        for key in used_values:
            inputs[key] = used_values[key]
        return inputs
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        chip_path = None,
        chip_names = None,
        layer_num = None,
        embedding_dim = None,
        prune_model = False,
        **kwargs,
    ):
        base_model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, **kwargs)
        chiped_model = cls(base_model, layer_num, embedding_dim)

        if (chip_path is not None):
            chiped_model.load_chips(chip_path, chip_names)

        if (prune_model):
            chiped_model.prune_model()
        return chiped_model
    
    def generate(
        self,
        **kwargs,
    ):
        _ = kwargs.pop('output_hidden_states', True)
        return super().generate(
            output_hidden_states=True,
            **kwargs,
        )