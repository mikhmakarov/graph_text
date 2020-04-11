import numpy as np
from sentence_transformers import SentenceTransformer


class SBert(object):
    def __init__(self):
        self.model = SentenceTransformer('bert-base-nli-stsb-mean-tokens')

    # fake fit to be consistent with count and tfidf vectorizers usage
    def fit_transform(self, texts):
        sentence_embeddings = self.model.encode(texts)

        return np.array(sentence_embeddings)
