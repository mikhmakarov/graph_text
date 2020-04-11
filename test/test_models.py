from models import TADW, TriDnr
from datasets import Cora, CiteseerM10
from text_transformers import SBert, LDA, W2V, Sent2Vec, Doc2Vec
from sklearn.feature_extraction.text import CountVectorizer


def test_tadw():
    ds = CiteseerM10()
    transformer = CountVectorizer()
    # transformer = SBert()
    # transformer = LDA()
    # transformer = W2V()
    # transformer = Sent2Vec()
    # transformer = Doc2Vec()
    ds.transform_features(transformer)
    data = ds.get_data()
    # model = TriDnr(data['graph'], data['features'], data['labels'])
    model = TADW(data['graph'], data['features'])
    model.learn_embeddings()
    print(123)
