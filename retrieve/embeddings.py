
import csv

import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.metrics.pairwise import pairwise_distances, pairwise_kernels

from retrieve.sparse_utils import set_threshold


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
    """
    Convenience class to handle embeddings. This class is better initialized
    from the method `from_csv`

    Arguments
    =========
    keys : list of strings representing the words in the rows of vectors
    vectors : an np.array(n_words, dim_size)
    """
    def __init__(self, keys, vectors):
        if len(keys) != len(vectors):
            raise ValueError("Expected {} vectors".format(len(keys)))
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
        """
        Arguments
        =========

        path : str, path to file with embeddings in csv format
            (word is assumed to go in first column)

        vocab : obtional, subset of words to load

        Output
        ======

        keys : dict, mapping words to the index in indices respecting the
            order in which the keys appear
        indices : list, mapping keys to the index in embedding matrix
        """
        df = load_embeddings(path, vocab=vocab)
        return cls(list(df.keys()), np.array(df).T)

    def get_indices(self, words):
        keys, indices = {}, []
        for idx, w in enumerate(words):
            if w in self.word2id:
                keys[w] = idx
                indices.append(self.word2id[w])
        return keys, indices

    def get_S(self, words=None, fill_missing=False,
              metric='cosine',
              beta=1, cutoff=0.0):
        """
        Arguments
        =========

        words : list (optional), words in desired order. The output matrix will
            have word-similarities ordered according to the order in `words`.
            However, if `fill_missing` is False, while the order is mantained,
            there will be gaps.

        fill_missing : bool, whether to fill similarities with one-hot vectors
            for out-of-vocabulary words

        Output
        ======
        keys : list of words ordered as the output matrix
        S : np.array (or scipy.sparse.lil_matrix) (vocab x vocab), this will be
            a sparse array if a positive `cutoff` is passed

        >>> vectors = [[0.35, 0.75], [0.5, 0.5], [0.75, 0.35]]
        >>> embs = Embeddings(['a', 'c', 'e'], np.array(vectors))
        >>> words = ['c', 'd', 'a', 'f']
        >>> S = embs.get_S(words=words, fill_missing=True)
        >>> S.shape             # asked for 4 words (fill_missing)
        (4, 4)
        >>> S[1, 3] == 0.0      # missing words evaluate to one-hot vectors
        True
        >>> w1, w2 = embs['a'], embs['c']
        >>> S[0, 2] == np.dot(w1, w2)/(np.linalg.norm(w1) * np.linalg.norm(w2))
        True
        >>> S[2, 0] == S[0, 2]
        True
        >>> keys, S = embs.get_S(words=words)
        >>> list(keys) == ['c', 'a']  # words only in keys in requested order
        True
        >>> S.shape             # only words in space (fill_missing=False)
        (2, 2)
        >>> w1, w2 = embs['a'], embs['c']
        >>> S[0, 1] == np.dot(w1, w2)/(np.linalg.norm(w1) * np.linalg.norm(w2))
        True
        """
        if fill_missing and not words:
            raise ValueError("`fill_missing` requires `words`")

        keys, indices = self.get_indices(words or self.keys)
        if not keys:
            raise ValueError("Couldn't find any of the requested words")

        S = pairwise_kernels(self.vectors[indices], metric=metric)
        # apply modifications on S
        S = np.power(np.clip(S, a_min=0, a_max=np.max(S)), beta)

        # drop elements
        if cutoff > 0.0:
            S = set_threshold(S, cutoff)

        # add one-hot vectors for OOV and rearrange to match input vocabulary
        if fill_missing:
            S_ = scipy.sparse.lil_matrix((len(words), len(words)))
            # rearrange
            src_x, src_y = np.meshgrid(indices, indices)
            keys2words = np.array([keys[w] for w in words if w in keys])
            trg_x, trg_y = np.meshgrid(keys2words, keys2words)
            S_[trg_x, trg_y] = S[src_x, src_y]
            S = S_
            # make sure diagonal is always 1
            S.setdiag(1)

            return S.tocsr()

        return keys, S

    def nearest_neighbours(self, words, metric='cosine', n=10, **kwargs):
        keys, index = self.get_indices(words)
        D = pairwise_distances(
            self.vectors[index], self.vectors, metric=metric, n_jobs=-1)
        # get neighbours
        neighs = np.argsort(D, axis=1)[:, 1: n+1]
        # get distances
        D = D[np.arange(len(keys)).repeat(n), np.ravel(neighs)]
        D = D.reshape(len(keys), -1)
        # human form
        neighs = [{self.id2word[neighs[i, j]]: D[i, j] for j in range(n)}
                  for i in range(len(keys))]

        return keys, neighs


def train_gensim_embeddings():
    pass


def export_fasttext_embeddings(path, vocab, output_path=None):
    try:
        import fastText
    except ModuleNotFoundError:
        raise ValueError("Couldn't import `fastText` module")

    model = fastText.load(path)
    keys, vectors = {}, []
    for idx, word in enumerate(vocab):
        keys[word] = idx
        vectors.append(model.get_word_vector(word))

    if output_path is not None:
        with open(output_path) as f:
            for word in keys:
                vec = ["{:.6}".format(i) for i in vectors[keys[word]].tolist()]
                f.write(word + '\t' + ' '.join(vec) + '\n')

    return keys, vectors


def evaluate_embeddings(eval_path):
    pass


if __name__ == '__main__':
    embs = Embeddings.from_csv('latin.lemma.embeddings')
    keys = ['anima', 'caput', 'manus']
    metric = 'cosine'
    n = 10
    nn_keys, neighs = embs.nearest_neighbours(keys, metric='euclidean')
    S = pairwise_distances(embs.vectors[:100], metric='cosine')
    D = pairwise_kernels(embs.vectors[:100], metric='cosine')
