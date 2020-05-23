from .dataset import Dataset
import pathlib

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()


class ScopusEcon(Dataset):
    def __init__(self,
                 graph_path=CURRENT_DIR.joinpath('../data/scopus_econ/edges.txt'),
                 texts_path=CURRENT_DIR.joinpath('../data/scopus_econ/docs.txt'),
                 labels_path=CURRENT_DIR.joinpath('../data/scopus_econ/labels.txt')):
        super(ScopusEcon, self).__init__(graph_path, texts_path, labels_path, partial=False)
