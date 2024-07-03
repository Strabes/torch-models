"""Sentencepiece Tokenizer"""

from tokenizers import SentencePieceBPETokenizer
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union, Tuple, List
import os
import json

@dataclass
class SentencepieceConfig:
    vocab_size: int = 1_000
    min_frequency: int = 5
    show_progress: bool = True
    padding: bool = True
    pad_token: str ="[PAD]"
    cls_token: Union[str,None] = "[CLS]"
    unk_token: str ='[UNK]'
    truncation: bool = True
    model_max_length: int = 512
    special_tokens: list = field(default_factory=lambda: ['[PAD]','[CLS]','[UNK]'])
    fitted_: bool = False

class Sentencepiece:
    """Wrapper on Huggingface SentencePieceBPETokenizer"""
    def __init__(
        self,
        config):

        self.config = config

    def fit(self, text):
        self.tokenizer = SentencePieceBPETokenizer(unk_token=self.config.unk_token)
        if self.config.cls_token:
            self.tokenizer.post_processor = TemplateProcessing(
                single=f"{self.config.cls_token}:0 $A:0",
                special_tokens = [(t,i) for i,t in enumerate(self.config.special_tokens)]
            )
        self.tokenizer.train_from_iterator(
            text,
            vocab_size=self.config.vocab_size,
            min_frequency=self.config.min_frequency,
            show_progress=self.config.show_progress,
            special_tokens=self.config.special_tokens
        )
        self.fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer._tokenizer,
            model_max_length=self.config.model_max_length,
            padding=self.config.padding,
            pad_token=self.config.pad_token,
            cls_token = self.config.cls_token,
            truncation=self.config.truncation
        )
        self.config.fitted_=True

    def transform(self, text):
        if not self.config.fitted_:
            raise ValueError("This Sentencepiece model is not yet fit.")
        res = self.fast_tokenizer(
            text,
            truncation = self.config.truncation,
            padding = 'max_length',
            max_length = self.config.model_max_length,
            return_tensors = 'pt'
        )
        return res

    def save(self, dir_path):
        dir_path = Path(dir_path)
        if dir_path.is_file():
            raise ValueError(f"dir_path must be a directory, but {dir_path} is a file")
        if not dir_path.exists():
            os.system(f"mkdir -p {str(dir_path)}")
        with open(str(dir_path / "metadata.json"), "w") as f:
            json.dump(self.config.__dict__,f)
        self.fast_tokenizer.save_pretrained(str(dir_path / "tokenizer"))

    @classmethod
    def load_from_dir(cls, dir_path):
        dir_path = Path(dir_path)
        with open(str(dir_path / "metadata.json"), "r") as f:
            metadata = json.load(f)
        config = SentencepieceConfig(**metadata)
        sp = cls(config)
        sp.fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(dir_path / "tokenizer"),
            unk_token = sp.unk_token
        )
        return sp