class BaseModel(object):
    def __init__(self, graph: str, features: str):
        self.graph = graph
        self.features = features
        self.embeddings = None

    def learn_embeddings(self):
        pass
