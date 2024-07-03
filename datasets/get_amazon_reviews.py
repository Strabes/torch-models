from datasets import load_dataset
import pandas as pd
from datetime import datetime
import torch
from torch_models import PACKAGE_ROOT

DOWNLOAD_DATA = False
if DOWNLOAD_DATA:
    reviews_dataset = load_dataset(
        "McAuley-Lab/Amazon-Reviews-2023",
        'raw_review_Books',
        split = 'full',
        streaming=True,
        trust_remote_code=True)
    reviews_dataset_head = reviews_dataset.take(10_000)

    data = (pd.DataFrame([i for i in reviews_dataset_head])
        .assign(verified_purchase = lambda df: df.verified_purchase.astype(str))
        .assign(timestamp = lambda df: df.timestamp.map(
            lambda x: datetime.utcfromtimestamp(x/1000)))
        .assign(month = lambda df: df.timestamp.dt.strftime("%B"))
        .assign(day = lambda df: df.timestamp.dt.strftime("%A"))
        .assign(n_images = lambda df: df.images.map(len)))
    data.to_pickle(str(PACKAGE_ROOT.parents[1] / "datasets/amazon_reviews.pkl"))
else:
    data = pd.read_pickle(str(PACKAGE_ROOT.parents[1] / "datasets/amazon_reviews.pkl"))

from torch_models.preprocess.preprocess import Preprocessor, PreprocessorConfig
from torch_models.tokenizers.sentencepiece import SentencepieceConfig

tokenizer_params = {
    'title': {
        'tokenizer_type': 'Sentencepiece',
        'sentencepiece_config': SentencepieceConfig(model_max_length=256)
    },
    'text': {
        'tokenizer_type': 'Sentencepiece',
        'sentencepiece_config': SentencepieceConfig(model_max_length=256)
    }
}

preprocessor_config = PreprocessorConfig(   
    numeric_cols=['helpful_vote','n_images'],
    categorical_cols=['verified_purchase','month','day'],
    tokenizer_params=tokenizer_params)

preprocessor = Preprocessor(preprocessor_config)

preprocessor.fit(data)

res = preprocessor.transform(data.head(50))

from torch_models.models.basic_input_layer import BasicInputLayerConfig, BasicInputLayer

basic_input_layer_config = BasicInputLayerConfig(
    numeric_cols=preprocessor.numeric_pipeline.get_feature_names_out().tolist(),
    categorical_cols=preprocessor.categorical_pipeline.get_feature_names_out().tolist(),
    text_cols=['title','text'],
    text_token_cardinalities=(1000,1000),
    text_padding_index=(0,0),
    text_cols_max_tokens=(256,256),
    text_embedding_dim=8
)

basic_input_layer = BasicInputLayer(basic_input_layer_config)

obel = basic_input_layer(
    torch.from_numpy(res['numeric']).to(dtype=torch.float64),
    torch.from_numpy(res['categorical']).to(dtype=torch.float64),
    [v['input_ids'] for v in res['text'].values()])



from torch_models.models.base_convolutional_model import BaseConvolutionalModelConfig, BaseConvolutionalModel



basic_conv_config = BaseConvolutionalModelConfig(
    numeric_cols=preprocessor.numeric_pipeline.get_feature_names_out().tolist(),
    categorical_cols=preprocessor.categorical_pipeline.get_feature_names_out().tolist(),
    text_cols=['title','text'],
    text_token_cardinalities=(1000,1000),
    text_padding_index=(0,0),
    text_cols_max_tokens=(256,256),
    text_embedding_dim=8
)

base_model = BaseConvolutionalModel(basic_conv_config)

model_output = base_model(
    torch.from_numpy(res['numeric']).to(dtype=torch.float64),
    torch.from_numpy(res['categorical']).to(dtype=torch.float64),
    [v['input_ids'] for v in res['text'].values()])


from torch_models.models.base_transformer_model import BaseTransformerModelConfig, BaseTransformerModel

basic_transformer_config = BaseTransformerModelConfig(
    numeric_cols=preprocessor.numeric_pipeline.get_feature_names_out().tolist(),
    categorical_cols=preprocessor.categorical_pipeline.get_feature_names_out().tolist(),
    text_cols=['title','text'],
    text_token_cardinalities=(1000,1000),
    text_padding_index=(0,0),
    text_cols_max_tokens=(256,256),
    text_embedding_dim=8
)

base_transformer_model = BaseTransformerModel(basic_transformer_config)

transformer_model_output = base_transformer_model(
    torch.from_numpy(res['numeric']).to(dtype=torch.float64),
    torch.from_numpy(res['categorical']).to(dtype=torch.float64),
    [v['input_ids'] for v in res['text'].values()])