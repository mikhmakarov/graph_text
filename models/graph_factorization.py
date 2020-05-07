import numpy as np
import pandas as pd
import os
import shutil

from pathlib import Path
from datetime import datetime
from subprocess import run

from gem.embedding.gf import GraphFactorization

from models.base_model import BaseModel

CURRENT_DIR = Path(__file__).parent.absolute()


class GF(BaseModel):
    def __init__(self, graph, features, labels=None, dim=80):
        super(GF, self).__init__(graph, features, dim, labels)

    def learn_embeddings(self):
        model = GraphFactorization(d=self.dim, max_iter=10000, eta=1*10**-4, regu=1.0, data_set='ds')
        model.learn_embedding(self.graph)

        self.embeddings = model.get_embedding()


