class BaseModel(object):
    def __init__(self, graph_path: str, features_path: str):
        self.graph_path = graph_path
        self.features_path = features_path
        self.embeddings = None

    def learn_embeddings(self):
        pass
