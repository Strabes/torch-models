import torch
from torch import nn
from typing import List, Tuple
#from pydantic import BaseModel
from dataclasses import dataclass

@dataclass
class BasicInputLayerConfig:
    numeric_cols: Tuple[str] = ()
    categorical_cols: Tuple[str] = ()
    text_cols: Tuple[str] = ()
    text_token_cardinalities: Tuple[int] = ()
    text_padding_index: Tuple[int] = ()
    text_cols_max_tokens: Tuple[int] = ()
    text_embedding_dim: int = 16
    text_embedding_max_norm: float = 5.0
    dropout: float = 0.1
    dtype: torch.dtype = torch.float64


class BasicInputLayer(nn.Module):
    def __init__(
        self,
        config: BasicInputLayerConfig):
        super(BasicInputLayer, self).__init__()
        self.config = config
        # Specify model components
        # Numeric
        if len(config.numeric_cols) > 0:
            self.batch_norm = nn.BatchNorm1d(len(config.numeric_cols), dtype=config.dtype)
        # text
        if len(config.text_cols) > 0:
            self.text_embeddings = nn.ModuleList([
                nn.Embedding(
                    num_embeddings = num_embeddings,
                    embedding_dim = config.text_embedding_dim,
                    padding_idx = padding_idx,
                    max_norm = config.text_embedding_max_norm,
                    dtype=config.dtype)
                for num_embeddings, padding_idx in
                zip(config.text_token_cardinalities, config.text_padding_index)
            ])
    
    def forward(self, numeric, categorical, text):
        if len(self.config.numeric_cols) > 0:
            numeric = self.batch_norm(numeric)
        if len(self.config.text_cols) > 0:
            text = [self.text_embeddings[i](t) for i,t in enumerate(text)]
        return numeric, categorical, text