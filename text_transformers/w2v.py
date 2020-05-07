import numpy as np
import gensim.downloader as api

from gensim.models import Word2Vec

from utils import preprocess_text
from text_transformers.base_text_transformer import BaseTextTransformer


class W2V(BaseTextTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = None

    # fake fit to be consistent with count and tfidf vectorizers usage
    def fit_transform(self, texts):
        clean_texts = [preprocess_text(t) for t in texts]
        if self.train:
            size = self.d
            self.model = Word2Vec(size=size, workers=8, min_count=3)
            self.model.build_vocab(clean_texts)
            self.model.train(clean_texts, total_examples=len(clean_texts), epochs=10)
        else:
            size = 300
            self.model = api.load('word2vec-google-news-300')

        embs = []
        for text in clean_texts:
            emb = np.zeros(size)
            n = 0
            for w in text:
                if w in self.model:
                    emb += self.model[w]
                    n += 1

            if n != 0:
                emb = emb / n

            embs.append(emb)

        sentence_embeddings = np.array(embs)

        return sentence_embeddings
