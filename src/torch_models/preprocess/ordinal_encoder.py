"""ReindexedOrdinalEncoder"""

import numpy as np
from sklearn.preprocessing import OrdinalEncoder


class ReindexedOrdinalEncoder(OrdinalEncoder):
    def __init__(
        self,
        *,
        categories="auto",
        sep_unknown_missing=True,
        min_frequency=None,
        max_categories=None,
    ):
        self.categories = categories
        self.dtype = int
        self.handle_unknown = "use_encoded_value"
        self.sep_unknown_missing = sep_unknown_missing
        self.unknown_value = -1
        if sep_unknown_missing:
            self.encoded_missing_value = -2
        else:
            self.encoded_missing_value = -1
        self.min_frequency = min_frequency
        self.max_categories = max_categories

    def fit(self, X, y=None):
        super(ReindexedOrdinalEncoder, self).fit(X, y)
        return self

    def transform(self, X):
        transformed = super(ReindexedOrdinalEncoder, self).transform(X)
        return transformed + (2 if self.sep_unknown_missing else 1)
    
    def inverse_transform(self, X):
        super(ReindexedOrdinalEncoder, self).inverse_transform(X - 1)

    @property
    def cardinality(self):
        cardinality = {}
        if hasattr(self, "feature_names_in_"):
            idx = self.feature_names_in_
        else:
            idx = range(self.n_features_in_)
        for i, feature in enumerate(idx):
            cardinality[feature] = (
                # number of distinct values (minus one if np.nan is one of the values)
                len([i for i in self.categories_[i] if i is not np.nan]) 
                # number of distinct values that are getting mapped to infrequent value
                - (0 if self.infrequent_categories_[i] is None 
                   else len(self.infrequent_categories_[i]) - 1) 
                # number of additional
                + (2 if self.sep_unknown_missing else 1))
        return cardinality

if __name__ == '__main__':
    import pandas as pd
    roe = ReindexedOrdinalEncoder(min_frequency = 0.2)
    df = pd.DataFrame({
        'x': ['a']*5 + ['b']*20 + ['c']*10 + ['d']*4 + [np.nan] + [None],
        'y': ['a']*20 + ['f']*10 + ['g']*10 + [np.nan]
    })
    roe.fit(df)
    print(roe.categories_)
    print(roe.infrequent_categories_)
    print(roe.cardinality)
    a = roe.transform(pd.concat([df,pd.DataFrame({'x':['e'],'y':['z']})]))
    print(np.unique(a[:,0]))
    print(np.unique(a[:,1]))