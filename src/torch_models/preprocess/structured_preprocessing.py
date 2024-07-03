from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch_models.preprocess.ordinal_encoder import ReindexedOrdinalEncoder
import pandas as pd
from sklearn.impute import SimpleImputer
from typing import Union, List, Dict

def create_numeric_pipeline(
    params: Union[Dict, None] = None,
    cols: List[str] = []):

    pipeline = Pipeline([
        ("arb_num_imputer", SimpleImputer(
            strategy='constant',
            fill_value=0,
            missing_values=pd.NA,
            add_indicator=True
        )),
        ("standard_scaler", StandardScaler())
    ])

    pipeline = ColumnTransformer([('num', pipeline, cols)], remainder='drop')

    if params:
        pipeline.set_params(**params)
    return pipeline

def _create_categorical_ordinal_encoder_pipeline(
    params: Union[Dict, None] = None,
    cols: List[str] = []):

    pipeline = Pipeline([
        ("cat_imputer", SimpleImputer(
            strategy='constant',
            fill_value='_MISSING_',
            missing_values=pd.NA
        )),
        ("reindexed_ordinal_encoder", ReindexedOrdinalEncoder(
            min_frequency = 0.02,
            max_categories = 20
        ))
    ])

    pipeline = ColumnTransformer([('cat',pipeline,cols)], remainder='drop')

    if params:
        pipeline.set_params(**params)
    
    return pipeline

def _create_categorical_one_hot_encoder_pipeline(
    params: Union[Dict, None] = None,
    cols: List[str] = []):

    pipeline = Pipeline([
        ("cat_imputer", SimpleImputer(
            strategy='constant',
            fill_value='_MISSING_',
            missing_values=pd.NA
        )),
        ("one_hot_encoder", OneHotEncoder(
            min_frequency = 0.02,
            max_categories = 20,
            sparse_output = False
        ))
    ])

    pipeline = ColumnTransformer([('cat',pipeline,cols)], remainder='drop')

    if params:
        pipeline.set_params(**params)
    
    return pipeline

def create_categorical_pipeline(
    categorical_encoding,
    params: Union[Dict, None] = None,
    cols: List[str] = []):
    if categorical_encoding == 'ordinal':
        return _create_categorical_ordinal_encoder_pipeline(params,cols)
    elif categorical_encoding == 'onehot':
        return _create_categorical_one_hot_encoder_pipeline(params,cols)
    else:
        raise ValueError(
        f"""`categorical_encoding` must be either 'ordinal' or 'onehot'
        but got {categorical_encoding}""")