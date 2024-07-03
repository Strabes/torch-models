import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple
#from pydantic import BaseModel
from dataclasses import dataclass

from torch_models.models.basic_input_layer import (
    BasicInputLayerConfig, BasicInputLayer)

@dataclass
class BaseConvolutionalModelConfig(BasicInputLayerConfig):
    text_convolution_filter_sizes: Tuple[int] = (4,8,16)
    text_convolution_num_filters: Tuple[int] = (100,100,100)
    text_output_layer_dim: int = 16
    hidden_layer_dims: Tuple[int] = (64,)

class BaseConvolutionalModel(nn.Module):
    def __init__(
        self,
        config: BaseConvolutionalModelConfig):
        super(BaseConvolutionalModel, self).__init__()
        self.config = config

        self.basic_input_layer = BasicInputLayer(config)
        if len(config.text_cols) > 0:
            self.conv1d_list = [nn.ModuleList([
                nn.Conv1d(
                    in_channels = config.text_embedding_dim,
                    out_channels = i,
                    kernel_size = j,
                    dtype = config.dtype)
                    for i,j in zip(
                        config.text_convolution_num_filters,
                        config.text_convolution_filter_sizes)
            ]) for t in config.text_cols]

            self.text_output_layers = nn.ModuleList([
                nn.Linear(
                    np.sum(config.text_convolution_num_filters),
                    config.text_output_layer_dim,
                    dtype = config.dtype
                ) for t in config.text_cols
            ])

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
        text_len = len(self.config.text_cols) * self.config.text_output_layer_dim
        return numeric_len + categorical_len + text_len

    def forward(self, numeric, categorical, text):

        numeric, categorical, text = self.basic_input_layer(numeric, categorical, text)

        if len(self.config.text_cols) > 0:
            text_conv_list = []
            for i, t in enumerate(text):
                t = t.permute(0,2,1)
                x_conv_list = [F.relu(conv1d(t)) for conv1d in self.conv1d_list[i]]
                x_pool_list = [F.max_pool1d(x_conv, kernel_size = x_conv.shape[2])
                    for x_conv in x_conv_list]
                x = torch.cat([x_pool.squeeze(dim=2) for x_pool in x_pool_list], dim=1)
                text_conv_list.append(self.text_output_layers[i](x))

        items_to_concat = []
        if len(self.config.numeric_cols) > 0:
            items_to_concat.append(numeric)
        if len(self.config.categorical_cols) > 0:
            items_to_concat.append(categorical)
        if len(self.config.text_cols) > 0:
            items_to_concat += text_conv_list
        t = torch.cat(items_to_concat, dim=1)
        for i in range(len(self.config.hidden_layer_dims)):
            t = F.relu(t)
            t = self.dropout_list[i](t)
            t = self.hidden_layers[i](t)
        
        return t