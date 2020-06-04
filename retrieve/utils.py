
import csv

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances, pairwise_kernels


def get_ngrams(s, min_n=1, max_n=1, sep='--'):
    """
    N-gram generator over input sequence. Allows multiple n-gram orders at once.
    """
    for n in range(min_n, max_n + 1):
        for ngram in zip(*[s[i:] for i in range(n)]):
            yield sep.join(ngram)


def chunks(it, size):
    """
    Chunk a generator into a given size (last chunk might be smaller)
    """
    buf = []
    for s in it:
        buf.append(s)
        if len(buf) == size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def load_stopwords(path):
    """
    Load stopwords from vertical format file. Ignore comments (#) and (?) doubted
    """
    stopwords = []
    with open(path) as f:
        for line in f:
            if line.startswith('?') or line.startswith('#'):
                continue
            stopwords.append(line.strip())
    return set(stopwords)


def load_freqs(path, top_k=0):
    """
    Load frequencies from file in format:

        word1 123
        word2 5
    """
    freqs = {}
    with open(path) as f:
        for line in f:
            count, w = line.strip().split()
            freqs[w] = int(count)
            if top_k > 0 and len(freqs) >= top_k:
                break
    return freqs


def load_embeddings(path, vocab=None):
    embs = pd.read_csv(
        path, sep=" ", header=None, index_col=0, skiprows=0, quoting=csv.QUOTE_NONE)
    embs = embs.dropna(axis=1, how='all')
    embs = embs.T
    if vocab is not None:
        # drop words not in vocab
        embs.drop(embs.columns.difference(vocab), 1, inplace=True)
    return embs


class Embeddings:
    def __init__(self, keys, vectors):
        self.word2id = {}
        self.id2word = {}
        for idx, word in enumerate(keys):
            self.word2id[word] = idx
            self.id2word[idx] = word
        self.vectors = vectors

    def __getitem__(self, key):
        return self.vectors[self.word2id[key]]

    @property
    def keys(self):
        return dict(self.word2id)

    def get_vectors(self, keys):
        targets = [w for w in keys if w in self.word2id]
        return targets, np.array(self[w] for w in targets)

    def default_vector(self):
        return np.mean(self.vectors, 0)

    @classmethod
    def from_csv(cls, path, vocab=None):
        df = load_embeddings(path, vocab=vocab)
        return cls(list(df.keys()), np.array(df).T)

    def get_indices(self, words):
        keys, indices = [], []
        for w in words:
            if w in self.word2id:
                keys.append(w)
                indices.append(self.word2id[w])
        return keys, indices

    def get_S(self, words=None, metric='cosine', fill_missing=False):
        """
        fill_missing : bool, whether to fill similarities with one-hot vectors
            for out-of-vocabulary words
        """
        keys, indices = self.get_indices(words or self.keys)
        if words and fill_missing:
            S = np.zeros((len(words), len(words)))
            x, y = np.meshgrid(indices, indices)
            S[x, y] = pairwise_kernels(self.vectors[indices], metric='cosine')
        else:
            S = pairwise_kernels(self.vectors[indices], metric='cosine')

        # make sure diagonal is always 1
        np.fill_diagonal(S, 1)

        return keys, S

    def nearest_neighbours(self, words, metric='cosine', n=10, **kwargs):
        keys, index = self.get_indices(words)
        D = pairwise_distances(
            self.vectors[index], self.vectors, metric=metric, njobs=-1)
        # get neighbours
        neighs = np.argsort(D, axis=1)[:, 1: n+1]
        # get distances
        D = D[np.arange(len(keys)).repeat(n), np.ravel(neighs)]
        D = D.reshape(len(keys), -1)
        # human form
        neighs = [{self.id2word[neighs[i, j]]: D[i, j] for j in range(n)}
                  for i in range(len(keys))]

        return keys, neighs


if __name__ == '__main__':
    embs = Embeddings.from_csv('../patrology/latin.token.embeddings')
    keys = ['anima', 'caput', 'manus']
    metric = 'cosine'
    n = 10
    embs.nearest_neighbours(keys, metric='euclidean')
    S = pairwise_distances(embs.vectors[:100], metric='cosine')
    D = pairwise_kernels(embs.vectors[:100], metric='cosine')
