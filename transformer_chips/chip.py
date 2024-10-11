from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ChipConfig:
    task: str
    chip_type: str
    in_dim: int
    out_dim: int
    hidden_dim: int = -1
    layer: int = -1


class Chip(nn.Module):
    def __init__(
            self,
            config: ChipConfig = None,
            **kwargs,
        ):
        super(Chip, self).__init__()
        self.config = config
        if (config is None):
            return

        if (config.chip_type == 'linear'):
            self.model = nn.Linear(config.in_dim, config.out_dim)
        elif (config.chip_type == '2xMLP'):
            self.model = nn.Sequential(
                nn.Linear(config.in_dim, config.hidden_dim),
                nn.ReLU(),
                nn.Linear(config.hidden_dim, config.out_dim),
            )
        else:
            self.model = None

        self.criterion = kwargs.pop('criterion', F.cross_entropy)
        self.probe_pos = kwargs.pop('probe_pos', -1)

    def set_config(self, config):
        self.config = config

    def get_name(self):
        return f"{self.config.task}.{self.config.chip_type}.{self.config.layer}"

    @staticmethod
    def build_name(task, chip_type, layer):
        return f"{task}.{chip_type}.{layer}"

    @classmethod
    def from_dict(cls, config_dict: dict):
        config = ChipConfig(
            task=config_dict['task'],
            chip_type=config_dict['chip_type'],
            in_dim=config_dict['in_dim'],
            out_dim=config_dict['out_dim'],
            hidden_dim=config_dict.get('hidden_dim', -1),
            layer=config_dict['layer'],
        )
        return cls(config)

    @classmethod
    def from_model(cls, model):
        chip = cls(config=None)
        chip.model = model
        return chip

    def forward(self, inputs, probe_pos=None):
        if (probe_pos is None):
            embeds = inputs[:, self.probe_pos, :]
        else:
            embeds = inputs[torch.arange(inputs.size(0)), probe_pos]
        return self.model(embeds)
    

class GenerationChip(Chip):
    def __init__(self, config: ChipConfig = None):
        super(GenerationChip, self).__init__(config)
        self.projector = nn.Linear(config.in_dim, config.in_dim)
        self.lm_head = nn.Linear(config.in_dim, config.out_dim)

    def forward(self, inputs, norm_function=None):
        projection = self.projector(inputs) + inputs
        if (norm_function is not None):
            projection = norm_function(projection)
        return self.lm_head(projection)