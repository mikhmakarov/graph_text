import os
import gensim.models as g
import numpy as np


CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

#parameters
model = os.path.join(CURRENT_PATH, "enwiki_dbow/doc2vec.bin")
test_docs="./test_docs.txt"
output_file="./test_vectors.txt"

#inference hyper-parameters
start_alpha = 0.01
infer_epoch = 1000

# #load model
# m = g.Doc2Vec.load(model)
# test_docs = [ x.strip().split() for x in codecs.open(test_docs, "r", "utf-8").readlines() ]
#
# #infer test vectors
# output = open(output_file, "w")
# for d in test_docs:
#     output.write(" ".join([str(x) for x in m.infer_vector(d, alpha=start_alpha, steps=infer_epoch)]) + "\n" )
# output.flush()
# output.close()


class Doc2Vec(object):
    def __init__(self):
        self.model = g.Doc2Vec.load(model)

    # fake fit to be consistent with count and tfidf vectorizers usage
    def fit_transform(self, texts):
        sentence_embeddings = []
        for d in texts:
            emb = self.model.infer_vector(d.split(), alpha=start_alpha, steps=infer_epoch)
            sentence_embeddings.append(emb)

        sentence_embeddings = np.array(sentence_embeddings)

        return sentence_embeddings
