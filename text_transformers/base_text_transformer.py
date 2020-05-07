from abc import ABC, abstractmethod
from typing import List

class BaseTextTransformer(ABC):
    def __init__(self, train=False, d=100):
        self.train = train
        self.d = d

    @abstractmethod
    def fit_transform(self, texts: List[str]):
        pass
