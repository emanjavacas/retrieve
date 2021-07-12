
import math
import logging

import numpy as np
import scipy.sparse
from joblib import Parallel, delayed, effective_n_jobs

import tqdm

from ...sparse_utils import sparse_chunks, set_threshold


logger = logging.getLogger(__name__)


def soft_cosine_simple(query, index, S):
    if scipy.sparse.issparse(query):
        query = dict(query.todok())
    if scipy.sparse.issparse(index):
        index = dict(index.todok())

    num = den1 = den2 = 0
    for _, i in set(query).union(index):
        for _, j in set(query).union(index):
            sim = S[i, j]
            num  += sim * query.get((0, i), 0) * index.get((0, j), 0)
            den1 += sim * query.get((0, i), 0) * query.get((0, j), 0)
            den2 += sim * index.get((0, i), 0) * index.get((0, j), 0)

    if den1 == 0 or den2 == 0:
        return 0.0

    return (num / (math.sqrt(den1) * math.sqrt(den2)))


def soft_cosine_similarities(queries, index, S, chunk_size=500, threshold=0.0,
                             disable_bar=False):
    """
    This function assumes that the order of vocabulary in the similarity matrix
    correspondes to the order of the vocabulary in the document representations

    Arguments
    =========

    queries : np.array(n, vocab), n query docs in BOW format, possibly sparse
    index : np.array(m, vocab), m indexed docs in BOW format, possibly sparse
    S : np.array(vocab, vocab), similarity matrix (possibly raised to a power)
        possibly a sparse matrix
    chunk_size : int, specifies the number of rows in queries to be computed
        at once. Helpful for handling memory constraints
    threshold : float, sparsify any output cells below such threshold

    Output
    ======
    sims : np.array(n, m), soft cosine similarities
    """
    is_S_sparse = scipy.sparse.issparse(S)

    (n, _), (m, _) = queries.shape, index.shape
    sims = scipy.sparse.lil_matrix((n, m)) if is_S_sparse else np.zeros((n, m))

    # (vocab x m) assumes m << n, so this can be stored in memory
    SindexT = S @ index.T
    # (m x vocab) x (m x vocab) -> (m x vocab)
    den2 = index.multiply(SindexT.T)
    # (m x vocab) -> (m)
    den2 = np.squeeze(np.array(den2.sum(1)))
    den2 = np.sqrt(den2)

    n_chunks = queries.shape[0] // chunk_size
    for (i_start, i_stop), Q in tqdm.tqdm(sparse_chunks(queries, chunk_size),
                                          total=n_chunks, desc='Soft cosine',
                                          disable=disable_bar):
        # (chunk_size x vocab) @ (vocab x m) -> (chunk_size x m)
        num = Q @ SindexT

        # (vocab x vocab) x (vocab x chunk_size) -> (vocab x chunk_size)
        SqueryT = S @ Q.T
        # (chunk_size x vocab) x (chunk_size x vocab) -> (chunk_size x vocab)
        den1 = Q.multiply(SqueryT.T)
        # (chunk_size x vocab) -> (chunk_size)
        den1 = np.squeeze(np.array(den1.sum(1)))
        den1 = np.sqrt(den1)

        # (chunk_size x m)
        Q_sims = (den1[:, None] * den2[None, :])
        with np.errstate(divide='ignore'):
            # this might run into division by 0
            Q_sims = num.multiply(1 / Q_sims) if is_S_sparse else (num / Q_sims)

        Q_sims = np.nan_to_num(Q_sims, copy=False)

        # apply threshold
        set_threshold(Q_sims, threshold)

        sims[i_start:i_stop, :] = Q_sims

    if scipy.sparse.isspmatrix_lil(sims):
        sims = sims.tocsr()

    return sims


def _parallel_soft_cosine1(index, S):
    # (vocab x m) assumes m << n, so this can be stored in memory
    SindexT = S @ index.T
    # (m x vocab) x (m x vocab) -> (m x vocab)
    den2 = index.multiply(SindexT.T)
    # (m x vocab) -> (m)
    den2 = np.squeeze(np.array(den2.sum(1)))
    den2 = np.sqrt(den2)

    return SindexT, den2


def _parallel_soft_cosine2(Q, SindexT, S, den2, is_S_sparse):
    # (chunk_size x vocab) @ (vocab x m) -> (chunk_size x m)
    num = Q @ SindexT

    # (vocab x vocab) x (vocab x chunk_size) -> (vocab x chunk_size)
    SqueryT = S @ Q.T
    # (chunk_size x vocab) x (chunk_size x vocab) -> (chunk_size x vocab)
    den1 = Q.multiply(SqueryT.T)
    # (chunk_size x vocab) -> (chunk_size)
    den1 = np.squeeze(np.array(den1.sum(1)))
    den1 = np.sqrt(den1)

    # (chunk_size x m)
    Q_sims = (den1[:, None] * den2[None, :])
    with np.errstate(divide='ignore'):
        # this might run into division by 0
        Q_sims = num.multiply(1 / Q_sims) if is_S_sparse else (num / Q_sims)

    Q_sims = np.nan_to_num(Q_sims, copy=False)

    return Q_sims


def wrapper_fn_assign(sims, i_start, i_stop, threshold, fn, *args, **kwargs):
    out = fn(*args, **kwargs)
    set_threshold(out, threshold)
    sims[i_start:i_stop] = out


def get_chunks(M, n_chunks):
    n, _ = M.shape
    chunk_size = n // n_chunks
    for i in range(n_chunks):
        start = i * chunk_size
        stop = min((i + 1) * chunk_size, n)
        yield start, stop
    if stop < n:
        yield stop, n


def parallel_soft_cosine(queries, index, S, threshold=0.0, n_jobs=-1):
    """
    Same as soft_cosine_similarities but with parallelized computations

    Arguments
    =========

    queries : np.array(n, vocab), n query docs in BOW format, possibly sparse
    index : np.array(m, vocab), m indexed docs in BOW format, possibly sparse
    S : np.array(vocab, vocab), similarity matrix (possibly raised to a power),
        possibly a sparse matrix
    threshold : float, sparsify any output cells below such threshold
    n_jobs : int, number of jobs to run in parallel

    Output
    ======
    sims : np.array(n, m), soft cosine similarities
    """
    is_S_sparse = scipy.sparse.issparse(S)

    (n, _), (m, _) = queries.shape, index.shape

    SindexT, den2 = _parallel_soft_cosine1(index, S)

    # following sklearn/metrics/pairwise
    fd = delayed(wrapper_fn_assign)
    sims = scipy.sparse.lil_matrix((n, m)) if is_S_sparse else np.zeros((n, m))

    n_chunks = effective_n_jobs(n_jobs)
    logger.info("Using {} CPUs".format(n_chunks))
    Parallel(backend='threading', n_jobs=n_chunks)(
        fd(sims, i_start, i_stop, threshold,
           # fn
           _parallel_soft_cosine2,
           # args
           queries[i_start: i_stop], SindexT, S, den2, is_S_sparse)
        for i_start, i_stop in get_chunks(queries, n_chunks))

    if scipy.sparse.isspmatrix_lil(sims):
        sims = sims.tocsr()

    return sims


if __name__ == '__main__':
    import timeit

    from retrieve.corpora import load_vulgate
    from retrieve.data import Criterion, TextPreprocessor, FeatureSelector
    from retrieve import utils
    from retrieve.embeddings import Embeddings
    from retrieve.methods import Tfidf

    # load
    vulg = load_vulgate(include_blb=True)
    # preprocess
    TextPreprocessor(
        stopwords=utils.Stopwords('data/stop/latin_berra.stop')
    ).process_collections(vulg, min_n=1, max_n=2, sep=' ')
    # drop features and get vocabulary
    fsel = FeatureSelector(vulg)
    vocab = fsel.filter_collections(
        vulg, criterion=(Criterion.DF >= 2) & (Criterion.FREQ >= 5))
    # get documents
    feats = vulg.get_features()
    # transform to tfidf
    tfidf = Tfidf(vocab).fit(feats)
    feats = tfidf.transform(feats)
    query, index = feats[:10000], feats[10000:20000]
    vocab = [ft for ft in tfidf.vectorizer.get_feature_names() if len(ft.split()) == 1]
    # load embeddings, make sure S is in same order as vocab
    embs = Embeddings.from_resource(
        'latin.lemma.ft.dim100.mc2.embeddings.gz', vocab=vocab)

    for cutoff in [0.25, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95]:
        S = embs.get_S(
            vocab=tfidf.vectorizer.get_feature_names(),
            fill_missing=True, cutoff=cutoff)
        nelem = S.shape[0] ** 2
        total_sim_size = (nelem - S.shape[0]) / 2
        effec_sim_size = (S.nnz - S.shape[0]) / 2
        print(" - Sparse run for threshold {:.2f}".format(cutoff))
        print("     - Off-diagonal pairs {}: {:.2f}%".format(
            int(effec_sim_size), effec_sim_size / total_sim_size))

        a = timeit.Timer(lambda: soft_cosine_similarities(
            query, index, S, threshold=0.25, chunk_size=100, disable_bar=True)
        ).timeit(number=5)
        b = timeit.Timer(lambda: soft_cosine_similarities(
            query, index, S.todense(), threshold=0.25, chunk_size=100, disable_bar=True)
        ).timeit(number=5)
        print("   - Runtime sparse: {:.3f}".format(a))
        print("   - Runtime dense: {:.3f}".format(b))

        a = timeit.Timer(lambda: parallel_soft_cosine(
            query, index, S, threshold=0.25)
        ).timeit(number=5)
        b = timeit.Timer(lambda: parallel_soft_cosine(
            query, index, S.todense(), threshold=0.25)
        ).timeit(number=5)
        print("   - Runtime parallel sparse: {:.3f}".format(a))
        print("   - Runtime parallel dense: {:.3f}".format(b))

    # - Sparse run for threshold 0.25
    #     - Off-diagonal pairs 4238493: 0.33%
    #   - Runtime sparse: 40.134
    #   - Runtime dense: 3.659
    # - Sparse run for threshold 0.50
    #     - Off-diagonal pairs 264663: 0.02%
    #   - Runtime sparse: 7.002
    #   - Runtime dense: 3.718
    # - Sparse run for threshold 0.70
    #     - Off-diagonal pairs 4877: 0.00%
    #   - Runtime sparse: 2.678
    #   - Runtime dense: 3.770
    # - Sparse run for threshold 0.80
    #     - Off-diagonal pairs 556: 0.00%
    #   - Runtime sparse: 2.666
    #   - Runtime dense: 3.621
    # - Sparse run for threshold 0.85
    #     - Off-diagonal pairs 209: 0.00%
    #   - Runtime sparse: 2.658
    #   - Runtime dense: 3.719
    # - Sparse run for threshold 0.90
    #     - Off-diagonal pairs 65: 0.00%
    #   - Runtime sparse: 2.507
    #   - Runtime dense: 3.672
    # - Sparse run for threshold 0.95
    #     - Off-diagonal pairs 2: 0.00%
    #   - Runtime sparse: 2.553
    #   - Runtime dense: 3.604
