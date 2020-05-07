import numpy as np
import pandas as pd
import os
import shutil

from pathlib import Path
from datetime import datetime
from subprocess import run

from models.base_model import BaseModel

CURRENT_DIR = Path(__file__).parent.absolute()
NODE2VEC_PATH = '/Users/mikhail-makarov/hse/year_2/thesis/node2vec/src/main.py'


class Node2Vec(BaseModel):
    def __init__(self, graph, features, labels=None, dim=80):
        super(Node2Vec, self).__init__(graph, features, dim, labels)

    def __dump_graph(self, directory):
        edgelist_path = str(directory.joinpath('edges.txt'))

        with open(edgelist_path, 'w') as edges_file:
            for edge in self.graph.edges():
                edges_file.write(f"{edge[0]} {edge[1]}\n")

        return edgelist_path

    def learn_embeddings(self):
        timestamp = str(datetime.now()).replace(' ', 'T')
        directory_path = CURRENT_DIR.joinpath('node2vec_' + timestamp)
        Path(directory_path).mkdir()
        edgelist_path = self.__dump_graph(directory_path)

        embeddings_file = str(CURRENT_DIR.joinpath('node2vec_embeddings_' + timestamp))
        node2vec_command = f"python {NODE2VEC_PATH} --input {edgelist_path} " \
                           f"--output {embeddings_file}  --workers 8 " \
                           f"--dimensions {self.dim} --num-walks 80 --window-size 10"

        call = 'zsh -c "source ~/.zshrc && conda activate node2vec && {}"'.format(node2vec_command)
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


