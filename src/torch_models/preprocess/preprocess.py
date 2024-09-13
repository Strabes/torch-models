"""Combined Preprocessor"""

from torch_models.preprocess.structured_preprocessing import (
    create_numeric_pipeline,
    create_categorical_pipeline
)
from torch_models.tokenizers.sentencepiece import Sentencepiece
from torch_models.tokenizers.bert_wordpiece import BERTWordpiece
import torch
from typing import Union, List, Dict, Tuple
import os
from pathlib import Path
import pickle
import json
from dataclasses import dataclass, field

@dataclass
class PreprocessorConfig:
    numeric_params: Union[Dict,None] = None
    numeric_cols: Tuple[str] = ()
    categorical_params: Union[Dict, None] = None
    categorical_cols: Tuple[str] = ()
    categorical_encoding: str = 'onehot'
    tokenizer_params: Union[Dict, None] = None
    fitted_: bool = False

TOKENIZERS = ['Sentencepiece', 'BERTWordpiece']

class Preprocessor:

    def __init__(
        self,
        config: PreprocessorConfig):

        self.config = config

        self.numeric_pipeline = create_numeric_pipeline(
            params = config.numeric_params,
            cols = config.numeric_cols
        )

        self.categorical_pipeline = create_categorical_pipeline(
            categorical_encoding=config.categorical_encoding,
            params = config.categorical_params,
            cols = config.categorical_cols
        )

        self.init_tokenizers()

    def fit(self, X, y=None):
        self.numeric_pipeline.fit(X,y)
        self.categorical_pipeline.fit(X,y)
        for col in self.tokenizers.keys():
            texts = X[col].fillna("").tolist()
            self.tokenizers[col].fit(texts)
        self.config.fitted_ = True
        return self

    def transform(self, X):
        numeric_output = self.numeric_pipeline.transform(X)
        categorical_output = self.categorical_pipeline.transform(X)
        text_cols = {}
        for col in self.tokenizers.keys():
            texts = X[col].fillna("").tolist()
            text_cols[col] = self.tokenizers[col].transform(texts)
        return {"numeric": numeric_output, "categorical": categorical_output, "text": text_cols}

    def init_tokenizers(self):
        self.tokenizers = {}
        if isinstance(self.config.tokenizer_params,dict):
            for k,v in self.config.tokenizer_params.items():
                self.tokenizers[k] = self._init_tokenizer(v)

    def _init_tokenizer(self, params):
        if params["tokenizer_type"] not in TOKENIZERS:
            raise ValueError(f"""Tokenizer type needs to be one of
            {", ".join(TOKENIZERS)} but got {params["tokenizer_type"]}""")
        if params["tokenizer_type"] == "Sentencepiece":
            return Sentencepiece(params["config"])
        elif params["tokenizer_type"] == "BERTWordpiece":
            return BERTWordpiece(params["config"])

    def save(self, dir_path):
        dir_path = Path(dir_path)
        if dir_path.is_file():
            raise ValueError(f"""{str(dir_path)} is a file but needs to be
            a directory""")
        if not dir_path.exists():
            os.system(f"mkdir -p {str(dir_path)}")
        with open(str(dir_path / "metadata.json"), "w") as f:
            json.dump(self.config.__dict__, f)
        with open(str(dir_path / "numeric_pipeline.pkl"), "wb") as f:
            pickle.dump(self.numeric_pipeline,f)
        with open(str(dir_path / "categorical_pipeline.pkl"), "wb") as f:
            pickle.dump(self.categorical_pipeline,f)
        for k,v in self.tokenizers.items():
            v.save(str(dir_path / k))