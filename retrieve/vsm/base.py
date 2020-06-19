
import inspect

from sklearn.metrics.pairwise import pairwise_kernels
import scipy.sparse
import tqdm

from retrieve.compare import soft_cosine_similarities
from retrieve.sparse_utils import sparse_chunks, set_threshold


def init_sklearn_vectorizer(vectorizer, vocab=None, **kwargs):
    params = set(inspect.signature(vectorizer).parameters)
    return vectorizer(
        vocabulary=list(vocab),
        # ensure l2 norm to speed up cosine similarity
        norm='l2',
        # overwrite default to avoid ignoring input
        token_pattern=r'\S+',
        **{k: v for k, v in kwargs.items() if k in params})


class VSM:
    def __init__(self, *args, **kwargs):
        """
        Dummy constructor, since not all methods require initalization
        """
        pass

    def fit(self, *args, **kwargs):
        """
        Dummy fit method (not compulsory)
        """
        return self

    def get_similarities(self, queries, index,
                         threshold=0.25, metric='linear',
                         chunk_size=-1, disable_bar=False, **kwargs):
        """
        Get similarities according to specified kernel.

        Arguments
        =========

        threshold : float (optional), discard similarities below threshold,
            helps enforcing sparsity of the resulting matrix

        metric : str, (optional), default is "linear" which corresponds to
            a cosine similarity since all `retrieve` vectorizers output
            normalized vectors

        chunk_size : int, (optional), if greater than 0, the query matrix will
            be partitioned in chunk of `chunk_size` items and processed in batches.
            In combination with a certain `threshold`, this will allow to fit
            the whole (sparse) similarity matrix in memory
        """
        index, queries = list(index), list(queries)
        transform = self.transform(queries + index)
        queries, index = transform[:len(queries)], transform[len(queries):]

        if chunk_size > 0:
            (n, _), (m, _) = queries.shape, index.shape
            sims = scipy.sparse.lil_matrix((n, m))
            n_chunks = n // chunk_size
            for (i_start, i_stop), Q in tqdm.tqdm(sparse_chunks(queries, chunk_size),
                                                  total=n_chunks,
                                                  desc='Chunked similarities',
                                                  disable=disable_bar):
                Q_sims = pairwise_kernels(Q, index, metric=metric, n_jobs=-1)
                if threshold > 0:
                    Q_sims = set_threshold(Q_sims, threshold)
                    sims[i_start: i_stop, :] = Q_sims

        else:
            sims = pairwise_kernels(queries, index, metric=metric, n_jobs=-1)
            if threshold > 0.0:
                sims = set_threshold(sims, threshold)

        if scipy.sparse.isspmatrix_lil(sims):
            sims = sims.tocsr()

        return sims


class VSMSoftCosine(VSM):
    def __init__(self, vocab, vectorizer, **kwargs):
        self.vectorizer = init_sklearn_vectorizer(vectorizer, vocab=vocab, **kwargs)

    def get_soft_cosine_similarities(self, queries, index, embs, threshold=0.25,
                                     chunk_size=500, disable_bar=False, **kwargs):
        """
        Compute soft cosine similarities between queries and index using the
        (possibly sparse) similarity matrix S indexing the similarity between
        words i and j

        Arguments
        =========

        embs : Embeddings class that will be used to extract the similarity matrix
            for words in the inputs using the instantiated vectorizer

        chunk_size : int, (optional), if greater than 0, the query matrix will
            be partitioned in chunk of `chunk_size` items and processed in batches.
            In combination with a certain `threshold`, this will allow to fit
            the whole (sparse) similarity matrix in memory

        kwargs : extra arguments passed to embs.get_S
        """
        index, queries = list(index), list(queries)
        transform = self.transform(index + queries)
        index, queries = transform[:len(index)], transform[len(index):]
        S = embs.get_S(words=self.vectorizer.get_feature_names(),
                       fill_missing=True, **kwargs)
        sims = soft_cosine_similarities(
            queries, index, S, chunk_size=chunk_size, threshold=threshold,
            disable_bar=disable_bar)

        return sims
