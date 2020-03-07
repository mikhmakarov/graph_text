from .dataset import Dataset
import pathlib

CURRENT_DIR = pathlib.Path(__file__).parent.absolute()


class Cora(Dataset):
    def __init__(self,
                 graph_path=CURRENT_DIR.joinpath('../data/cora/cora.cites'),
                 texts_path=CURRENT_DIR.joinpath('../data/cora/cora.text'),
                 labels_path=CURRENT_DIR.joinpath('../data/cora/cora.labels')):
        super(Cora, self).__init__(graph_path, texts_path, labels_path)
