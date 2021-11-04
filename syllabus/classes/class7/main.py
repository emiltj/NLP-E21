import numpy as np
import torch

from datasets import load_dataset
import gensim.downloader as api

<<<<<<< HEAD
=======
from util import batch
from LSTM import RNN
from embedding import gensim_to_torch_embedding

>>>>>>> e4732e9599727118ad40b72a55bf0aceda78a541
# DATASET
dataset = load_dataset("conllpp")
train = dataset["train"]

# inspect the dataset
train["tokens"][:1]
train["ner_tags"][:1]
classes = train.features["ner_tags"].feature
num_classes = train.features["ner_tags"].feature.num_classes
classes
num_classes

# CONVERTING EMBEDDINGS
<<<<<<< HEAD
import numpy as np
import torch

=======
>>>>>>> e4732e9599727118ad40b72a55bf0aceda78a541
model = api.load("glove-wiki-gigaword-50")
type(model)
model.vectors.shape

# convert gensim word embedding to torch word embedding
embedding_layer, vocab = gensim_to_torch_embedding(model)

embedding_layer
vocab

# PREPARING A BATCH
<<<<<<< HEAD
def tokens_to_idx(tokens: List[str], vocab: dict = model.key_to_index) -> List[int]:
=======

# shuffle dataset
shuffled_train = dataset["train"].shuffle(seed=1)

# batch it using a utility function (don't spend time on the function, but make sure you understand the output)
batch_size = 10
batches_tokens = batch(shuffled_train["tokens"], batch_size)
batches_tags = batch(shuffled_train["ner_tags"], batch_size)


def tokens_to_idx(tokens, vocab=model.key_to_index):
>>>>>>> e4732e9599727118ad40b72a55bf0aceda78a541
    """
    Ideas to understand this function:
    - Write documentation for this function including type hints for each arguement and return statement
    - What does the .get method do?
    - Why lowercase?
    """
    return [vocab.get(t.lower(), vocab["UNK"]) for t in tokens]

# Example of function
asda = train["tokens"][:1][0]
asda.append("ajskd")
asda
tokens_to_idx(asda)

# sample using only the first batch
batch_tokens = next(batches_tokens)
batch_tags = next(batches_tags)
batch_tok_idx = [tokens_to_idx(sent) for sent in batch_tokens]
batch_size = len(batch_tokens)

# compute length of longest sentence in batch
batch_max_len = max([len(s) for s in batch_tok_idx])

# prepare a numpy array with the data, initializing the data with 'PAD'
# and all labels with -1; initializing labels to -1 differentiates tokens
# with tags from 'PAD' tokens
batch_input = vocab["PAD"] * np.ones((batch_size, batch_max_len))
batch_labels = -1 * np.ones((batch_size, batch_max_len))

# copy the data to the numpy array
for i in range(batch_size):
    tok_idx = batch_tok_idx[i]
    tags = batch_tags[i]
    size = len(tok_idx)

    batch_input[i][:size] = tok_idx
    batch_labels[i][:size] = tags

<<<<<<< HEAD
# since all data are indices, we convert them to torch LongTensors
=======

# since all data are indices, we convert them to torch LongTensors (integers)
>>>>>>> e4732e9599727118ad40b72a55bf0aceda78a541
batch_input, batch_labels = torch.LongTensor(batch_input), torch.LongTensor(
    batch_labels
)

# CREATE MODEL
model = RNN(
    embedding_layer=embedding_layer, output_dim=num_classes + 1, hidden_dim_size=256
)

# FORWARD PASS
X = batch_input
y = model(X)

loss = model.loss_fn(outputs=y, labels=batch_labels)
# loss.backward()