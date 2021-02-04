import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

from text_transformers.base_text_transformer import BaseTextTransformer


def iter_batches(iterable: list, n: int = 1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


class Ernie(BaseTextTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-en")
        self.model = AutoModel.from_pretrained("nghuyong/ernie-2.0-en")

    # fake fit to be consistent with count and tfidf vectorizers usage
    def fit_transform(self, texts, batch_size=100):
        # sentence_embeddings = self.model.encode(texts)

        batch_embs = []
        for text_batch in tqdm(iter_batches(texts.tolist(), batch_size)):
            max_length = min(max([len(t.split(" ")) for t in text_batch]), 512)
            tokenized = self.tokenizer(text_batch,
                                       padding="max_length",
                                       max_length=max_length,
                                       truncation="only_first")["input_ids"]
            tokenized = torch.tensor(tokenized)
            out = self.model(tokenized)
            embeddings_of_last_layer = out[0]
            cls_embeddings = embeddings_of_last_layer[:,0,:].detach().numpy()
            batch_embs.append(cls_embeddings)

        return np.vstack(batch_embs)


if __name__ == "__main__":
    ernie = Ernie()
    texts = [
        "First test sentence",
         "Second something else",
        "Third completely different and with different size"
    ]
    ernie.fit_transform(texts)