import re
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from scipy import sparse

# nltk.download('wordnet')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('stopwords')

stop_words = stopwords.words('english')

lemmatizer = WordNetLemmatizer()
stemmer = SnowballStemmer('english')


def normalize_adjacency(graph):
    """
    Method to calculate a sparse degree normalized adjacency matrix.
    :param graph: Sparse graph adjacency matrix.
    :return A: Normalized adjacency matrix.
    """
    ind = range(len(graph.nodes()))
    degs = [1.0/graph.degree(node) for node in graph.nodes()]
    edges = [edge for edge in graph.edges()]
    index_1 = [edge[0] for edge in edges] + [edge[1] for edge in edges]
    index_2 = [edge[1] for edge in edges] + [edge[0] for edge in edges]
    values = [1.0 for _ in edges] + [1.0 for _ in edges]
    shape = (len(ind), len(ind))
    A = sparse.coo_matrix((values, (index_1, index_2)), shape=shape, dtype=np.float32)
    degs = sparse.coo_matrix((degs, (ind, ind)), shape=A.shape, dtype=np.float32)
    A = A.dot(degs)
    return A


def preprocess_text(text, stem=True):
    tokens = []
    text = re.sub(r'[^a-zA-Z \t]+', ' ', text)
    for t, pos in nltk.pos_tag(nltk.word_tokenize(text.lower())):
        if isinstance(pos, str):
            wntag = pos[0].lower()
            wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
        else:
            wntag = None

        if wntag is not None:
            lem = lemmatizer.lemmatize(t, wntag)
        else:
            lem = t

        if not lem.isalpha() or len(lem) < 3:
            continue

        if lem in stop_words:
            continue

        if stem:
            lem = stemmer.stem(lem)

        tokens.append(lem)

    return tokens
