
import numpy as np
import scipy.sparse
import numba as nb


@nb.njit()
def _top_k_dense(data, indices, indptr, k):
    # indptr holds pointers to indices and data
    # indices[indptr[0]:indptr[1]] -> index of nonzero items in 1st row
    # data[indptr[0]:indptr[1]]    -> nonzero items in 1st row
    nrows = indptr.shape[0] - 1  # substract one because of format

    # output variables
    top_indices = np.zeros((nrows, k), dtype=indices.dtype) - 1
    top_data = np.zeros((nrows, k), dtype=data.dtype) * np.nan

    for i in nb.prange(nrows):
        start, stop = indptr[i], indptr[i + 1]
        top_k = np.argsort(data[start:stop])[::-1][:k]
        n_items = min(len(top_k), k)
        # assign
        top_indices[i, 0:n_items] = indices[start:stop][top_k]
        top_data[i, 0:n_items] = data[start:stop][top_k]

    return top_indices, top_data


@nb.njit()
def _top_k_sparse_data(data, indices, indptr, k):
    # indptr holds pointers to indices and data
    # indices[indptr[0]:indptr[1]] -> index of nonzero items in 1st row
    # data[indptr[0]:indptr[1]]    -> nonzero items in 1st row
    nrows = indptr.shape[0] - 1  # substract one because of format

    top_indices = []
    top_vals = []
    top_rows = []
    top_cols = []

    for i in nb.prange(nrows):
        start, stop = indptr[i], indptr[i + 1]
        top_k = np.argsort(data[start:stop])[::-1][:k]
        if len(top_k) > 0:
            top_indices.append(indices[start:stop][top_k])
            top_vals.append(data[start:stop][top_k])
            top_cols.append(np.arange(0, len(top_k)))
            top_rows.append(np.repeat([i], len(top_k)))

    return top_indices, top_vals, top_rows, top_cols


def _top_k_sparse(data, indices, indptr, k):
    top_idxs, top_vals, rows, cols = _top_k_sparse_data(data, indices, indptr, k)
    top_idxs, top_vals = np.concatenate(top_idxs), np.concatenate(top_vals)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    top_idxs = scipy.sparse.csr_matrix((top_idxs, (rows, cols)), dtype=indices.dtype)
    top_vals = scipy.sparse.csr_matrix((top_vals, (rows, cols)), dtype=data.dtype)

    return top_idxs, top_vals


def top_k(X, k):
    """
    X : matrix, (n x m)

    Output
    ======

    np.array(n x k), top_k items per row, it doesn't exclude the
        highest one, which is in self-search typically corresponds
        to itself

    >>> # checkerboard pattern with rowise increments
    >>> nrow = 3
    >>> ncol = 5
    >>> X = scipy.sparse.dok_matrix((nrow, ncol))
    >>> for i in range(nrow):
    ...     for j in range(i % 2, ncol, 2):
    ...         X[i, j] = X.nnz + 1
    >>> indices, data = top_k(X, 2)
    >>> indices.tolist()
    [[4, 2], [3, 1], [4, 2]]
    >>> data.tolist()
    [[3.0, 2.0], [5.0, 4.0], [8.0, 7.0]]
    """
    if not scipy.sparse.issparse(X):
        return np.argsort(X, 1)[::-1][:, :k]

    X = X.tocsr()
    data, indices, indptr = X.data, X.indices, X.indptr
    return _top_k_dense(data, indices, indptr, k)


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
