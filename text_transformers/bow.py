import numpy as np

from sklearn.feature_extraction.text import CountVectorizer

from utils import preprocess_text
from text_transformers.base_text_transformer import BaseTextTransformer


class BOW(BaseTextTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_transform(self, texts):
        clean_texts = [' '.join(preprocess_text(t)) for t in texts]
        transformer = CountVectorizer(min_df=3, max_df=0.7)
        return np.array(transformer.fit_transform(clean_texts).todense())
