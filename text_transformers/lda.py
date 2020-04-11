import numpy as np
import gensim
import gensim.corpora as corpora
from utils import preprocess_text


class LDA(object):
    def __init__(self):
        pass

    # fake fit to be consistent with count and tfidf vectorizers usage
    def fit_transform(self, texts):
        clean_texts = [preprocess_text(t) for t in texts]
        id2word = corpora.Dictionary(clean_texts)
        id2word.filter_extremes(no_below=10, no_above=0.3)
        corpus = [id2word.doc2bow(text) for text in clean_texts]

        lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                               id2word=id2word,
                                               num_topics=20,
                                               alpha=0.1,
                                               eta=0.1,
                                               random_state=100,
                                               chunksize=100,
                                               passes=10,
                                               per_word_topics=True,
                                               minimum_probability=0.0)

        sentence_embeddings = lda_model[corpus]

        sentence_embeddings = np.array([[t[1] for t in r[0]] for r in sentence_embeddings])

        return sentence_embeddings
