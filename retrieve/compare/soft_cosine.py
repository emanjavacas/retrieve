
import math

import numpy as np
import scipy.sparse

import tqdm


def soft_cosine_simple(query, index, S):
    if scipy.sparse.issparse(query):
        query = query.todense()
    if scipy.sparse.issparse(index):
        index = index.todense()

    query, index = np.squeeze(np.array(query)), np.squeeze(np.array(index))
    num = den1 = den2 = 0
    for i in range(len(query)):
        for j in range(len(index)):
            sim = S[i, j]
            num  += sim * query[i] * index[j]
            den1 += sim * query[i] * query[j]
            den2 += sim * index[i] * index[j]

    return (num / (math.sqrt(den1) * math.sqrt(den2)))


def sparse_chunks(M, chunk_size):
    """
    This creates copy since sparse matrices don't have views
    """
    n, _ = M.shape
    for i in range(0, n, chunk_size):
        start, stop = i, min(i + chunk_size, n)
        yield (start, stop), M[start:stop]


def soft_cosine_similarities(queries, index, S, chunk_size=500):
    """
    This function assumes that the order of vocabulary in the similarity matrix
    correspondes to the order of the vocabulary in the document representations

    Arguments
    =========

    queries : np.array(n, vocab), n query docs in BOW format
    index : np.array(m, vocab), m indexed docs in BOW format
    S : np.array(vocab, vocab), similarity matrix (possibly raised to a power)

    Output
    ======
    sims : np.array(n, m), soft cosine similarities
    """
    is_S_sparse = scipy.sparse.issparse(S)
    n_chunks = queries.shape[0] // chunk_size

    (n, _), (m, _) = queries.shape, index.shape
    sims = scipy.sparse.lil_matrix((n, m)) if is_S_sparse else np.zeros((n, m))

    # (vocab x m) assumes m << n, so this can be stored in memory
    SindexT = S @ index.T
    # (m)
    den2 = index @ SindexT
    den2 = den2.diagonal() if is_S_sparse else np.diag(den2)
    den2 = np.sqrt(den2)

    for (i_start, i_stop), Q in tqdm.tqdm(sparse_chunks(queries, chunk_size),
                                          total=n_chunks, desc='Soft cosine'):
        # (chunk_size x vocab) @ (vocab x m) -> (chunk_size x m)
        num = Q @ SindexT

        # (chunk_size x vocab) @ (vocab x vocab) @ (vocab x chunk_size) -> chunk_size
        den1 = Q @ S @ Q.T
        den1 = den1.diagonal() if is_S_sparse else np.diag(den1)
        den1 = np.sqrt(den1)

        # (chunk_size x m)
        sims_Q = (den1[:, None] * den2[None, :])
        sims_Q = num.multiply(1/sims_Q) if is_S_sparse else (num / sims_Q)

        sims[i_start:i_stop, :] = sims_Q

    return np.nan_to_num(sims, copy=False)
