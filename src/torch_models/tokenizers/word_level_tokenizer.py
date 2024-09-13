from tokenizers.models import WordLevel as HFWordLevel
from tokenizers import Tokenizer
from tokenizers.trainers import WordLevelTrainer
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import Lowercase
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from pathlib import Path
import os
import json
from dataclasses import dataclass, field
from typing import Tuple, List, Union
import itertools
import re

@dataclass
class WordLevelConfig:
    vocab_size: int = 1_000
    min_frequency: int = 5
    show_progress: bool = True
    padding: bool = True
    pad_token: str ="[PAD]"
    cls_token: Union[str,None] = "[CLS]"
    unk_token: str ='[UNK]'
    limit_alphabet:int=72
    truncation: bool = True
    model_max_length: int = 512
    special_tokens: list = field(init=False)
    fitted_: bool = False

    def __post_init__(self):
        self.special_tokens = [self.pad_token, self.unk_token, self.cls_token]

class WordLevel:
    def __init__(
        self,
        config: WordLevelConfig
        ):
        self.config = config

    def fit(self, text):
        self.tokenizer = Tokenizer(HFWordLevel(unk_token=self.config.unk_token))
        self.tokenizer.normalizer = Lowercase()
        self.tokenizer.pre_tokenizer = Whitespace()
        if self.config.cls_token:
            self.tokenizer.post_processor = TemplateProcessing(
                single=f"{self.config.cls_token}:0 $A:0",
                special_tokens = [(t,i) for i,t in enumerate(self.config.special_tokens)])
        trainer = WordLevelTrainer(
            vocab_size=self.config.vocab_size,
            min_frequency = self.config.min_frequency,
            show_progress = self.config.show_progress,
            special_tokens = self.config.special_tokens
        )
        self.tokenizer.train_from_iterator(text, trainer=trainer)
        self.fast_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=self.tokenizer,
            model_max_length=self.config.model_max_length,
            padding=self.config.padding,
            pad_token=self.config.pad_token,
            unk_token=self.config.unk_token,
            cls_token = self.config.cls_token,
            truncation=self.config.truncation
        )
        self.config.fitted_=True

    def transform(self, text):
        if not self.config.fitted_:
            raise ValueError("This WordLevel model is not yet fit.")
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
        with open(str(dir_path / "config.json"), "w") as f:
            json.dump({k: self.config.__getattribute__(k)
                for k in self.config.__dataclass_fields__.keys()},f)
        self.fast_tokenizer.save_pretrained(str(dir_path / "tokenizer"))

    @classmethod
    def load_from_dir(cls, dir_path):
        dir_path = Path(dir_path)
        with open(str(dir_path / "config.json"), "r") as f:
            config = json.load(f)
        config = WordLevelConfig(**config)
        sp = cls(config)
        sp.fast_tokenizer = PreTrainedTokenizerFast.from_pretrained(
            str(dir_path / "tokenizer"),
            unk_token = sp.unk_token
        )
        return sp