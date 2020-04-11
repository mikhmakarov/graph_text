import numpy as np
import gensim.downloader as api
from utils import preprocess_text


class W2V(object):
    def __init__(self):
        self.model = api.load('word2vec-google-news-300')

    # fake fit to be consistent with count and tfidf vectorizers usage
    def fit_transform(self, texts):
        clean_texts = [preprocess_text(t) for t in texts]

        embs = []
        for text in clean_texts:
            emb = np.zeros(300)
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
