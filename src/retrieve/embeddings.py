
import time
import gzip
import csv
import logging
import sys
if sys.version_info.minor < 7:
    import importlib_resources
else:
    import importlib.resources as importlib_resources

import tqdm
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.metrics import pairwise_distances, pairwise_kernels

from retrieve.methods import pairwise_kernels_chunked
from retrieve.sparse_utils import top_k, substract_vector


logger = logging.getLogger(__name__)

SIM, DIST = 0, 1


def load_embeddings(path, vocab=None):
    # handle word2vec format
    skiprows = 0
    with open(path) as f:
        if len(next(f).strip().split()) == 2:
            skiprows = 1

    embs = pd.read_csv(
        path, sep=" ", header=None,
        index_col=0, skiprows=skiprows, quoting=csv.QUOTE_NONE)
    embs = embs.dropna(axis=1, how='all')
    embs = embs.T

    if vocab is not None:
        # drop words not in vocab
        missing = embs.columns.difference(vocab)
        logger.info("Dropping {} words from vocabulary".format(len(missing)))
        embs.drop(missing, 1, inplace=True)

    return embs


def load_fasttext(path):
    try:
        import fastText
        return fastText.load(path)
    except ModuleNotFoundError:
        try:
            import fasttext
            return fasttext.load_model(path)
        except ModuleNotFoundError:
            raise ValueError("Couldn't import `fastText` or `fasttext` module")


def normalize_vectors(vectors):
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


class Embeddings:
    # class constants
    SIM = SIM
    DIST = DIST

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

    def __len__(self):
        return len(self.word2id)

    def __getitem__(self, key):
        return self.vectors[self.word2id[key]]

    def __contains__(self, key):
        return key in self.word2id

    def apply_projection(self, proj, batch_size=4096, renorm=False):
        import torch
        if isinstance(proj, str):
            # assume path
            proj = torch.load(proj)
        vectors = self.vectors
        if renorm:
            vectors = normalize_vectors(vectors)
        for i in tqdm.tqdm(range(0, len(self.vectors), batch_size)):
            start, stop = i, min(i + batch_size, len(vectors))
            vectors[start:stop, :] = (proj @ vectors[start:stop, :].T).T
        self.vectors = vectors

    def normalize_vectors(self):
        self.vectors = normalize_vectors(self.vectors)

    @property
    def keys(self):
        return dict(self.word2id)

    def get_vectors(self, keys):
        targets = [w for w in keys if w in self.word2id]
        return targets, np.array(list(map(self.__getitem__, targets)))

    def default_vector(self):
        return np.mean(self.vectors, 0)

    @classmethod
    def require_embeddings(cls, embs, msg='', **kwargs):
        if isinstance(embs, str):
            embs = cls.from_file(embs, **kwargs)
        if not isinstance(embs, cls):
            raise ValueError(msg)
        return embs

    @classmethod
    def from_fasttext(cls, path, vocab):
        if vocab is None:
            raise ValueError("FastText model requires vocab")

        model = load_fasttext(path)

        vectors = []
        for word in vocab:
            vectors.append(model.get_word_vector(word))

        return cls(vocab, np.array(vectors))

    @classmethod
    def from_file(cls, path, vocab=None, skip_header=False):
        # dispatch fastText
        if path.endswith('bin'):
            return cls.from_fasttext(path, vocab)

        if vocab is not None:
            vocab = set(vocab)
            logger.info("Loading {} word embeddings".format(len(vocab)))
        keys, vectors = [], []

        open_fn = gzip.open if path.endswith(".gz") else open
        with open_fn(path) as f:
            if skip_header:
                next(f)
            for line in f:
                if isinstance(line, bytes):
                    line = line.decode()
                word, *vec = line.strip().split()
                if vocab and word not in vocab:
                    continue
                keys.append(word)
                vectors.append(np.array(vec, dtype=np.float))

        # report missing
        if vocab is not None:
            logger.info("Loaded {}/{} words from vocabulary".format(
                len(keys), len(vocab)))

        return cls(keys, np.array(vectors))

    @classmethod
    def from_csv(cls, path, vocab=None):
        """
        Arguments
        =========

        path : str, path to file with embeddings in csv format
            (word is assumed to go in first column)

        vocab : optional, subset of words to load

        Output
        ======

        keys : dict, mapping words to the index in indices respecting the
            order in which the keys appear
        indices : list, mapping keys to the index in embedding matrix
        """
        df = load_embeddings(path, vocab=vocab)
        return cls(list(df.keys()), np.array(df).T)

    @classmethod
    def from_resource(cls, path, vocab=None):
        if not importlib_resources.is_resource('retrieve.resources.misc', path):
            raise ValueError("Unknown resource: {}".format(path))
        with importlib_resources.path('retrieve.resources.misc', path) as f:
            return cls.from_file(str(f), vocab=vocab)

    def to_csv(self, path):
        with open(path, 'w') as f:
            for idx, word in sorted(self.id2word.items()):
                vec = ["{:.6}".format(i) for i in self.vectors[idx].tolist()]
                f.write(word + '\t' + ' '.join(vec) + '\n')

    def get_indices(self, words):
        keys, indices = {}, []
        for idx, w in enumerate(words):
            if w in self.word2id:
                keys[w] = idx
                indices.append(self.word2id[w])
        return keys, indices

    def get_S(self, vocab=None, fill_missing=False,
              metric='cosine', beta=1, apply_mod=True, cutoff=0.0, chunk_size=0):
        """
        Arguments
        =========

        vocab : list (optional), vocab in desired order. The output matrix will
            have word-similarities ordered according to the order in `vocab`.
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
        >>> vocab = ['c', 'd', 'a', 'f']
        >>> S = embs.get_S(vocab=vocab, fill_missing=True)
        >>> S.shape             # asked for 4 words (fill_missing)
        (4, 4)
        >>> S[1, 3] == 0.0      # missing words evaluate to one-hot vectors
        True
        >>> w1, w2 = embs['a'], embs['c']
        >>> sim = np.dot(w1, w2)/(np.linalg.norm(w1) * np.linalg.norm(w2))
        >>> np.allclose(S[0, 2], sim)
        True
        >>> S[2, 0] == S[0, 2]
        True
        >>> keys, S = embs.get_S(vocab=vocab)
        >>> list(keys) == ['c', 'a']  # words only in keys in requested order
        True
        >>> S.shape             # only words in space (fill_missing=False)
        (2, 2)
        >>> w1, w2 = embs['a'], embs['c']
        >>> sim = np.dot(w1, w2)/(np.linalg.norm(w1) * np.linalg.norm(w2))
        >>> np.allclose(S[0, 1], sim)
        True
        """
        if fill_missing and not vocab:
            raise ValueError("`fill_missing` requires `vocab`")
        if apply_mod and beta > 1 and cutoff is not None and cutoff < 0:
            raise ValueError("Negative cutoff with positive beta yields wrong results")

        keys, indices = self.get_indices(vocab or self.keys)
        if not keys:
            raise ValueError("Couldn't find any of the requested vocab")

        # (found words x found words)
        logger.info("Computing {} similarities".format(len(indices)))
        start = time.time()
        S = pairwise_kernels_chunked(
            self.vectors[indices], metric=metric, chunk_size=chunk_size,
            threshold=cutoff)
        logger.info("Got S in {:.2f} secs".format(time.time() - start))
        # apply modifications on S
        if apply_mod:
            S = (S.power(beta) if scipy.sparse.issparse(S) else np.power(S, beta))
        # add one-hot vectors for OOV and rearrange to match input vocabulary
        if fill_missing:
            # (requested words x requested words)
            S_ = scipy.sparse.lil_matrix((len(vocab), len(vocab)))
            # rearrange
            index = np.array([keys[w] for w in vocab if w in keys])
            index = np.tile(index, (len(index), 1))
            S_[index, index.T] = S
            S = S_
            # make sure diagonal is always 1
            S.setdiag(1)

            return S.tocsr()

        return keys, S

    def nearest_neighbours(self, words, n=10,
                           metric='cosine',
                           metric_type=SIM, csls_k=0):
        """
        If `metric_type` is Embeddings.SIM then `metric` must be one of
        sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS:
        - ['cosine', 'sigmoid', 'linear', etc.]
        If `metric_type` is Embeddings.DIST then `metric` must be one of
        sklearn.metrics.pairwise.PAIRWISE_DISTANCE_FUNCTIONS:
        - ['cosine', 'euclidean', 'l1', 'l2', 'manhattan', 'cityblock']
        """
        if csls_k > 0 and metric_type != SIM:
            raise ValueError("CSLS is defined over similarities not distances")

        keys, index = self.get_indices(words)

        if metric_type == Embeddings.SIM:
            if csls_k > 0:
                S = csls(
                    pairwise_kernels(self.vectors, metric=metric, n_jobs=-1),
                    csls_k)
                S = S[index]
            else:
                S = pairwise_kernels(
                    self.vectors[index], self.vectors, metric=metric, n_jobs=-1)
            # get neighbours
            neighs = np.argsort(-S, axis=1)[:, 1: n+1]
        elif metric_type == Embeddings.DIST:
            S = pairwise_distances(
                self.vectors[index], self.vectors, metric=metric, n_jobs=-1)
            neighs = np.argsort(S, axis=1)[:, 1: n+1]
        else:
            raise ValueError("Unknown `metric_type`")

        S = S[np.arange(len(keys)).repeat(n), np.ravel(neighs)]
        S = S.reshape(len(keys), -1)
        # human form
        neighs = [{self.id2word[neighs[i, j]]: S[i, j] for j in range(n)}
                  for i in range(len(keys))]

        return keys, neighs


def train_gensim_embeddings(path, output_path=None, **kwargs):
    from gensim.models import Word2Vec
    from gensim.models.word2vec import LineSentence
    from retrieve import enable_log_level

    m = Word2Vec(sentences=LineSentence(path), **kwargs)
    if output_path:
        m.wv.save_word2vec_format(output_path)

    return m


def export_fasttext_embeddings(path, vocab, output_path=None):
    model = load_fasttext(path)
    keys, vectors = {}, []
    for idx, word in enumerate(vocab):
        keys[word] = idx
        vectors.append(model.get_word_vector(word))

    if output_path is not None:
        with open(output_path, 'w+') as f:
            for word in keys:
                vec = ["{:.6}".format(i) for i in vectors[keys[word]].tolist()]
                f.write(word + '\t' + ' '.join(vec) + '\n')

    return keys, vectors


def csls_dense(S, k=10):
    indices, values = top_k(S, k + 1)
    mean_values = values[:, 1:].mean(1)
    return (2 * S) - mean_values[:, None] - mean_values[None, :]


def csls_sparse(S, k=10):
    indices, values = top_k(S, k + 1)
    mean_values = values[:, 1:].mean(1)
    S = substract_vector(
            substract_vector(S * 2, mean_values, axis=1),
                mean_values, axis=0)
    return S


def csls(S, k=10):
    if scipy.sparse.issparse(S):
        return csls_sparse(S, k)
    return csls_dense(S, k)


def csls_crosslingual(S, S1, S2, k=10):
    S1_mean = top_k(S1, k + 1)[1][:, 1:].mean(1)
    S2_mean = top_k(S2, k + 1)[1][:, 1:].mean(1)
    return (2 * S) - S1_mean[:, None] - S2_mean[None, :]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('--output')
    parser.add_argument('--size', type=int, default=100)
    parser.add_argument('--window', type=int, default=5)
    parser.add_argument('--min_count', type=int, default=1)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()

    m = train_gensim_embeddings(args.input, output_path=args.output,
                                size=args.size, window=args.window,
                                min_count=args.min_count, workers=args.workers)