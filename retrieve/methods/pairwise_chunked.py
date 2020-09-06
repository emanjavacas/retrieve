
import tqdm
import scipy.sparse
from sklearn.metrics.pairwise import pairwise_kernels

from ..sparse_utils import sparse_chunks, set_threshold


def pairwise_kernels_chunked(X, Y=None, metric='linear', chunk_size=0,
                             desc=None, disable_bar=True, threshold=None, n_jobs=-1):
    """
    Chunked version of pairwise_kernels that applies thresholds to produce
    sparse similarity matrices
    """
    if Y is None:
        Y = X

    if chunk_size > 0:
        (n, _), (m, _) = X.shape, Y.shape
        sims = scipy.sparse.lil_matrix((n, m))
        n_chunks = n // chunk_size
        for (i_start, i_stop), Q in tqdm.tqdm(sparse_chunks(X, chunk_size),
                                              total=n_chunks, desc=desc,
                                              disable=disable_bar):
            Q_sims = pairwise_kernels(Q, Y, metric=metric, n_jobs=-1)
            if threshold is not None:
                set_threshold(Q_sims, threshold)
                sims[i_start: i_stop, :] = Q_sims
    else:
        sims = pairwise_kernels(X, Y, metric=metric, n_jobs=n_jobs)
        if threshold is not None:
            set_threshold(sims, threshold)

    if scipy.sparse.isspmatrix_lil(sims):
        sims = sims.tocsr()

    return sims
