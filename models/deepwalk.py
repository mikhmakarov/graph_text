import numpy as np
import pandas as pd
import os
import shutil

from pathlib import Path
from datetime import datetime
from subprocess import run

from models.base_model import BaseModel

CURRENT_DIR = Path(__file__).parent.absolute()


class DeepWalk(BaseModel):
    def __init__(self, graph, features, labels=None, dim=80):
        super(DeepWalk, self).__init__(graph, features, dim, labels)

    def __dump_graph(self, directory):
        edgelist_path = str(directory.joinpath('edges.txt'))

        with open(edgelist_path, 'w') as edges_file:
            for edge in self.graph.edges():
                edges_file.write(f"{edge[0]} {edge[1]}\n")
                edges_file.write(f"{edge[1]} {edge[0]}\n")

        return edgelist_path

    def learn_embeddings(self):
        timestamp = str(datetime.now()).replace(' ', 'T')
        directory_path = CURRENT_DIR.joinpath('deepwalk_' + timestamp)
        Path(directory_path).mkdir()
        edgelist_path = self.__dump_graph(directory_path)

        embeddings_file = str(CURRENT_DIR.joinpath('deepwalk_embeddings_' + timestamp))
        deepwalk_command = f"deepwalk --format edgelist --input {edgelist_path} --output {embeddings_file} " \
                           f"--representation-size {self.dim} --number-walks 80 --window-size 10 --workers 8"
        call = 'zsh -c "source ~/.zshrc && conda activate deepwalk && {}"'.format(deepwalk_command)
        run(call, shell=True)

        embeddings_df = pd.read_csv(embeddings_file, sep=' ', header=None, skiprows=1,
                                    names=['id'] + [f'emb_{i}' for i in range(self.dim)])
        embeddings_df = embeddings_df.set_index('id').sort_index()

        for node in self.graph.nodes():
            # isolated node
            if node not in embeddings_df.index:
                embeddings_df.loc[node] = np.zeros(shape=self.dim)

        embeddings = embeddings_df.values

        shutil.rmtree(directory_path)
        os.remove(embeddings_file)

        self.embeddings = embeddings


