import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
from torch_models.models.positional_encoding import PositionalEncoding
#from pydantic import BaseModel
from dataclasses import dataclass

from torch_models.models.basic_input_layer import (
    BasicInputLayerConfig, BasicInputLayer)

@dataclass
class BaseTransformerModelConfig(BasicInputLayerConfig):
    text_transformer_model_dim: int = 8
    text_transformer_nhead: int = 2
    text_transformer_num_layers: int = 2
    text_transformer_dim_feedforward: int = 16
    text_output_dim: int = 8
    hidden_layer_dims: Tuple[int] = (64,)

class BaseTransformerModel(nn.Module):
    def __init__(
        self,
        config: BaseTransformerModelConfig):
        super(BaseTransformerModel, self).__init__()
        self.config = config

        self.basic_input_layer = BasicInputLayer(config)
        if len(config.text_cols) > 0:
            self.positional_encodings = nn.ModuleList([
                PositionalEncoding(config.text_embedding_dim, config.dropout, l) 
                for l in config.text_cols_max_tokens 
            ])
            self.text_transformer_encoder = nn.ModuleList([
                nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(
                        config.text_embedding_dim,
                        nhead = config.text_transformer_nhead,
                        batch_first=True,
                        dim_feedforward = config.text_transformer_dim_feedforward,
                        dtype = config.dtype
                    ),
                    num_layers=config.text_transformer_num_layers
                ) for i in config.text_cols
            ])
            self.text_linear_layer = nn.ModuleList([
                nn.Linear(config.text_embedding_dim, config.text_output_dim,dtype=config.dtype)
                for i in config.text_cols])

        module_list = []
        for i, dim in enumerate(config.hidden_layer_dims):
            if i == 0:
                module_list.append(
                    nn.Linear(
                        self.hidden_layer_input_dim, dim, dtype=config.dtype
                    )
                )
            else:
                module_list.append(
                    nn.Linear(
                        config.hidden_layer_dims[i-1], dim, dtype=config.dtype
                    )
                )
        self.hidden_layers = nn.ModuleList(module_list)
        self.dropout_list = nn.ModuleList([nn.Dropout(p=config.dropout) for i in config.hidden_layer_dims])

    @property
    def hidden_layer_input_dim(self):
        numeric_len = len(self.config.numeric_cols)
        categorical_len = len(self.config.categorical_cols)
        text_len = len(self.config.text_cols) * self.config.text_embedding_dim
        return numeric_len + categorical_len + text_len

    def forward(self, numeric, categorical, text):

        numeric, categorical, text = self.basic_input_layer(numeric, categorical, text)

        if len(self.config.text_cols) > 0:
            text = [self.positional_encodings[i](t) for i,t in enumerate(text)]
            text = [self.text_transformer_encoder[i](t) for i,t in enumerate(text)]
            text = [self.text_linear_layer[i](t[:,0,:]) for i,t in enumerate(text)]

        items_to_concat = []
        if len(self.config.numeric_cols) > 0:
            items_to_concat.append(numeric)
        if len(self.config.categorical_cols) > 0:
            items_to_concat.append(categorical)
        if len(self.config.text_cols) > 0:
            items_to_concat += text
        t = torch.cat(items_to_concat, dim=1)
        for i in range(len(self.config.hidden_layer_dims)):
            t = F.relu(t)
            t = self.dropout_list[i](t)
            t = self.hidden_layers[i](t)
        
        return t