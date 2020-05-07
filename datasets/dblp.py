from .dataset import Dataset
import pathlib

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()


class Dblp(Dataset):
    def __init__(self,
                 graph_path=CURRENT_DIR.joinpath('../data/dblp/edges.txt'),
                 texts_path=CURRENT_DIR.joinpath('../data/dblp/docs.txt'),
                 labels_path=CURRENT_DIR.joinpath('../data/dblp/labels.txt')):
        super(Dblp, self).__init__(graph_path, texts_path, labels_path)
