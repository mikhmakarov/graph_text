from .dataset import Dataset
import pathlib

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()


class CiteseerM10(Dataset):
    def __init__(self,
                 graph_path=CURRENT_DIR.joinpath('../data/citeseer_m10/edges.txt'),
                 texts_path=CURRENT_DIR.joinpath('../data/citeseer_m10/docs.txt'),
                 labels_path=CURRENT_DIR.joinpath('../data/citeseer_m10/labels.txt')):
        super(CiteseerM10, self).__init__(graph_path, texts_path, labels_path)
