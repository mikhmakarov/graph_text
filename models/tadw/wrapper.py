from ..base_model import BaseModel
from .helpers import read_graph, read_features
from .tadw import DenseTADW

from dotmap import DotMap
import numpy as np


class TadwModel(BaseModel):
    def __init__(self,
                 graph: str,
                 features: str,
                 order: int = 2,
                 dimensions: int = 32,
                 iterations: int = 200,
                 alpha: float = 10**-6,
                 lambd: float = 1000.0,
                 lower_control: float = 10**-15):
        super(TadwModel, self).__init__(graph, features)

        args = DotMap()
        args.order = order
        args.dimensions = dimensions
        args.iterations = iterations
        args.alpha = alpha
        args.lambd = lambd
        args.lower_control = lower_control



        self.ids = np.array(range(A.shape[0])).reshape(-1, 1)
        self.model = DenseTADW(A, X, args)

    def learn_embeddings(self):
        self.model.optimize()
        self.embeddings = self.model.compile_embedding(self.ids)


