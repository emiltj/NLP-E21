import numpy as np
import gensim.downloader as api
word_emb = api.load("glove-wiki-gigaword-50")

vocab = word_emb.index_to_key
dog_emb = word_emb["dog"]
cat_emb = word_emb["cat"]
graphite_emb = word_emb["graphite"]

# Attention exercises
# 1
np.dot(dog_emb, cat_emb)

# 2
np.dot(dog_emb, graphite_emb)

# 3
E = np.asmatrix([dog_emb, cat_emb, graphite_emb])
E.shape
E @ E.T
# The values in the resulting matrix corresponds to the dot product between all combinations.
# The dot product of vectors is a similarity measure (also called the projection)
# The diagonal is the dot product of the same embeddings, and therefore the length of this vector.

# 4
# Q (or K) is the hidden state vector from the decoder
# K (or Q) is the hidden state matrix from the encoder


