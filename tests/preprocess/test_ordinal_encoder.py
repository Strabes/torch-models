import pytest
import pandas as pd
import numpy as np

@pytest.fixture
def example_data():
    return pd.DataFrame({
        'x': ['a']*5 + ['b']*20 + ['c']*10 + ['d']*4 + [np.nan],
        'y': ['a']*20 + ['f']*10 + ['g']*10
    })

def test_ordinal_encoder():
    roe = ReindexedOrdinalEncoder(min_frequency = 0.2)