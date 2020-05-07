import numpy as np
import pandas as pd
import os
import shutil

from pathlib import Path
from datetime import datetime
from subprocess import run

from gem.embedding.hope import HOPE

from models.base_model import BaseModel

CURRENT_DIR = Path(__file__).parent.absolute()


class Hope(BaseModel):
    def __init__(self, graph, features, labels=None, dim=80):
        super(Hope, self).__init__(graph, features, dim, labels)

    def learn_embeddings(self):
        hope = HOPE(d=self.dim, beta=0.01)
        hope.learn_embedding(self.graph)

        self.embeddings = hope.get_embedding()


