import numpy as np
from sentence_transformers import SentenceTransformer
from text_transformers.base_text_transformer import BaseTextTransformer


class SBert(BaseTextTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

    # fake fit to be consistent with count and tfidf vectorizers usage
    def fit_transform(self, texts):
        sentence_embeddings = self.model.encode(texts)

        return np.array(sentence_embeddings)
