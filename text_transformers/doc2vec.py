import os
import gensim.models as g
import numpy as np

from gensim.models import Doc2Vec as Doc2VecGensim
from gensim.models.doc2vec import TaggedDocument


from text_transformers.base_text_transformer import BaseTextTransformer
from utils import preprocess_text


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

#parameters
model = os.path.join(CURRENT_PATH, "enwiki_dbow/doc2vec.bin")
test_docs="./test_docs.txt"
output_file="./test_vectors.txt"

#inference hyper-parameters
start_alpha = 0.01
infer_epoch = 1000


class Doc2Vec(BaseTextTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # self.model = g.Doc2Vec.load(model)

    # fake fit to be consistent with count and tfidf vectorizers usage
    def fit_transform(self, texts):
        clean_texts = [TaggedDocument(preprocess_text(t), tags='id_' + str(i)) for i, t in enumerate(texts)]
        if self.train:
            size = self.d
            self.model = Doc2VecGensim(size=size, workers=8, min_count=3)
            self.model.build_vocab(clean_texts)
            self.model.train(clean_texts, total_examples=len(clean_texts), epochs=10)
        else:
            self.model = g.Doc2Vec.load(model)

        sentence_embeddings = []
        for d in clean_texts:
            emb = self.model.infer_vector(d.words, alpha=start_alpha, steps=infer_epoch)
            sentence_embeddings.append(emb)

        sentence_embeddings = np.array(sentence_embeddings)

        return sentence_embeddings
