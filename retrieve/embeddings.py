
import csv
import logging

import pandas as pd
import numpy as np
import scipy.sparse

from retrieve.compare.pairwise_chunked import pairwise_kernels_chunked
from retrieve.sparse_utils import set_threshold


logger = logging.getLogger(__name__)


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
        logger.debug("Dropping {} words from vocabulary".format(len(missing)))
        embs.drop(missing, 1, inplace=True)

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
        return targets, np.array(list(map(self.__getitem__, targets)))

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
              metric='cosine', beta=1, cutoff=0.0, chunk_size=0):
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
        >>> sim = np.dot(w1, w2)/(np.linalg.norm(w1) * np.linalg.norm(w2))
        >>> np.allclose(S[0, 2], sim)
        True
        >>> S[2, 0] == S[0, 2]
        True
        >>> keys, S = embs.get_S(words=words)
        >>> list(keys) == ['c', 'a']  # words only in keys in requested order
        True
        >>> S.shape             # only words in space (fill_missing=False)
        (2, 2)
        >>> w1, w2 = embs['a'], embs['c']
        >>> sim = np.dot(w1, w2)/(np.linalg.norm(w1) * np.linalg.norm(w2))
        >>> np.allclose(S[0, 1], sim)
        True
        """
        if fill_missing and not words:
            raise ValueError("`fill_missing` requires `words`")

        keys, indices = self.get_indices(words or self.keys)
        if not keys:
            raise ValueError("Couldn't find any of the requested words")

        S = pairwise_kernels_chunked(
            self.vectors[indices], metric=metric, chunk_size=chunk_size)
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
        S = pairwise_kernels_chunked(
            self.vectors[index], self.vectors, metric=metric, n_jobs=-1)
        # get neighbours
        neighs = np.argsort(-S, axis=1)[:, 1: n+1]
        # get distances
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
        with open(output_path, 'w+') as f:
            for word in keys:
                vec = ["{:.6}".format(i) for i in vectors[keys[word]].tolist()]
                f.write(word + '\t' + ' '.join(vec) + '\n')

    return keys, vectors


def evaluate_embeddings(eval_path):
    pass


def cscl(X, Y, chunk_size):
    pass


def standard_nn(X, Z, batch_size):
    # translation = collections.defaultdict(int)
    # for i in range(0, len(src), BATCH_SIZE):
    #     j = min(i + BATCH_SIZE, len(src))
    #     similarities = x[src[i:j]].dot(z.T)
    #     nn = similarities.argmax(axis=1).tolist()
    #     for k in range(j-i):
    #         translation[src[i+k]] = nn[k]
    pass


def invnn(X, Z):
    # translation = collections.defaultdict(int)
    # best_rank = np.full(len(src), x.shape[0], dtype=int)
    # best_sim = np.full(len(src), -100, dtype=dtype)
    # for i in range(0, z.shape[0], BATCH_SIZE):
    #     j = min(i + BATCH_SIZE, z.shape[0])
    #     similarities = z[i:j].dot(x.T)
    #     ind = (-similarities).argsort(axis=1)
    #     ranks = asnumpy(ind.argsort(axis=1)[:, src])
    #     sims = asnumpy(similarities[:, src])
    #     for k in range(i, j):
    #         for l in range(len(src)):
    #             rank = ranks[k-i, l]
    #             sim = sims[k-i, l]
    #             if rank < best_rank[l] or (rank == best_rank[l] and sim > best_sim[l]):
    #                 best_rank[l] = rank
    #                 best_sim[l] = sim
    #                 translation[src[l]] = k
    pass


def invsoftmax(X, Z, inv_sample=None):
    # sample = xp.arange(x.shape[0]) if inv_sample is None else xp.random.randint(0, x.shape[0], inv_sample)
    # partition = xp.zeros(z.shape[0])
    # for i in range(0, len(sample), BATCH_SIZE):
    #     j = min(i + BATCH_SIZE, len(sample))
    #     partition += xp.exp(args.inv_temperature*z.dot(x[sample[i:j]].T)).sum(axis=1)
    # for i in range(0, len(src), BATCH_SIZE):
    #     j = min(i + BATCH_SIZE, len(src))
    #     p = xp.exp(args.inv_temperature*x[src[i:j]].dot(z.T)) / partition
    #     nn = p.argmax(axis=1).tolist()
    #     for k in range(j-i):
    #         translation[src[i+k]] = nn[k]
    pass


def csls(X, Z):
    # knn_sim_bwd = xp.zeros(z.shape[0])
    # for i in range(0, z.shape[0], BATCH_SIZE):
    #     j = min(i + BATCH_SIZE, z.shape[0])
    #     knn_sim_bwd[i:j] = topk_mean(z[i:j].dot(x.T), k=args.neighborhood, inplace=True)
    # for i in range(0, len(src), BATCH_SIZE):
    #     j = min(i + BATCH_SIZE, len(src))
    #     similarities = 2*x[src[i:j]].dot(z.T) - knn_sim_bwd  # Equivalent to the real CSLS scores for NN
    #     nn = similarities.argmax(axis=1).tolist()
    #     for k in range(j-i):
    #         translation[src[i+k]] = nn[k]
    pass


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
