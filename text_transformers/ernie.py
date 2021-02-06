import numpy as np
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

from text_transformers.base_text_transformer import BaseTextTransformer


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Going to train on {device}')


def iter_batches(iterable: list, n: int = 1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


class Ernie(BaseTextTransformer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained("nghuyong/ernie-2.0-en")
        self.model = AutoModel.from_pretrained("nghuyong/ernie-2.0-en").to(device)
        # print(summary(self.model, input_size=(1000, 512)))

    # fake fit to be consistent with count and tfidf vectorizers usage
    def fit_transform(self, texts, batch_size=16):
        # sentence_embeddings = self.model.encode(texts)

        with torch.no_grad():
            batch_embs = []
            for text_batch in tqdm(iter_batches(texts.tolist(), batch_size)):
                max_length = min(max([len(t.split(" ")) for t in text_batch]), 512)
                tokenized = self.tokenizer(text_batch,
                                           padding="max_length",
                                           max_length=max_length,
                                           truncation="only_first")["input_ids"]
                tokenized = torch.tensor(tokenized).to(device)
                out = self.model(tokenized)
                embeddings_of_last_layer = out[0]
                embeddings = embeddings_of_last_layer.mean(axis=1).detach().cpu().numpy()
                batch_embs.append(embeddings)

        return np.vstack(batch_embs)
