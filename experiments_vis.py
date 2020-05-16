from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

from models import TADW, TriDnr, DeepWalk, Node2Vec, Hope, GCN_Model
from text_transformers import SBert, LDA, W2V, Sent2Vec, Doc2Vec, BOW, TFIDF
from datasets import Cora, CiteseerM10, Dblp
from task import VisTask

dataset = Cora()

task = ('GCN (d=64)', VisTask(dataset, TFIDF, TADW, d=160, labels=True))


name, vis_task = task

embeddings, labels = vis_task.get_embeddings()

print(123)