import networkx as nx
import pandas as pd
import os
import shutil

from pathlib import Path
from datetime import datetime
from subprocess import run

from models.base_model import BaseModel

CURRENT_DIR = Path(__file__).parent.absolute()
TRIDNR_PATH = '/Users/mikhail-makarov/hse/year_2/thesis/TriDNR/learn_embeddings.py'


class TriDnr(BaseModel):
    def __init__(self, graph, features, labels, dim=80):
        super(TriDnr, self).__init__(graph, features, dim, labels)

    def __dump_graph(self, directory):

        with open(directory.joinpath('adjedges.txt'), 'w') as docs:
            for line in nx.generate_adjlist(self.graph):
                docs.write(f'{line}\n')

        texts = [t.replace('\n', ' ') for t in self.features]

        with open(directory.joinpath('docs.txt'), 'w') as docs:
            for _id, text in zip(self.graph.nodes(), texts):
                docs.write(f'{_id} {text}\n')

        with open(directory.joinpath('labels.txt'), 'w') as docs:
            for _id, label in zip(self.graph.nodes(), self.labels):
                docs.write(f'{_id} {label}\n')

    def learn_embeddings(self):
        timestamp = str(datetime.now()).replace(' ', 'T')
        directory_path = CURRENT_DIR.joinpath('tridnr_' + timestamp)
        Path(directory_path).mkdir()
        self.__dump_graph(directory_path)

        embeddings_file = str(CURRENT_DIR.joinpath('tridnr_embeddings_' + timestamp))
        tridnr_command = TRIDNR_PATH + ' ' + str(directory_path) + ' ' + embeddings_file + ' ' + str(self.dim)
        call = 'zsh -c "source ~/.zshrc && conda activate tridnr && python {}"'.format(tridnr_command)
        run(call, shell=True)

        embeddings = pd.read_csv(embeddings_file, header=None).values

        shutil.rmtree(directory_path)
        os.remove(embeddings_file)

        self.embeddings = embeddings


