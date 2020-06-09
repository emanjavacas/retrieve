
import numpy as np
import scipy.sparse
import numba as nb


def top_k():
    pass


def argsort():
    pass


def sparse_chunks(M, chunk_size):
    """
    This creates copy since sparse matrices don't have views
    """
    n, _ = M.shape
    for i in range(0, n, chunk_size):
        start, stop = i, min(i + chunk_size, n)
        yield (start, stop), M[start:stop]


def set_threshold(X, threshold, sparse_matrix=scipy.sparse.csr_matrix):
    if not scipy.sparse.issparse(X):
        X[np.where(X < threshold)] = 0.0
        return sparse_matrix(X)

    if isinstance(X, scipy.sparse.lil_matrix):
        raise ValueError("Cannot efficiently drop items on lil_matrix")

    X.data[np.abs(X.data) < threshold] = 0.0
    X.eliminate_zeros()
    return X
