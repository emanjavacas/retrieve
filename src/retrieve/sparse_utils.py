
import numpy as np
import scipy.sparse
import numba as nb


@nb.njit()
def substract_vector_loop(data, indices, indptr, vec):
    for i in nb.prange(vec.shape[0]):
        data[indptr[i]:indptr[i+1]] -= vec[i]
    return data, indices, indptr


def substract_vector(S, vec, inplace=False, axis=0):
    orig_type = type(S)
    X = S
    if not inplace:
        X = S.copy()
    if axis == 0:
        X = X.tocsr()
    elif axis == 1:
        X = X.tocsc()
    substract_vector_loop(X.data, X.indices, X.indptr, vec)
    return orig_type(X)


@nb.njit()
def _top_k_dense(data, indices, indptr, k):
    # indptr holds pointers to indices and data
    # indices[indptr[0]:indptr[1]] -> index of nonzero items in 1st row
    # data[indptr[0]:indptr[1]]    -> nonzero items in 1st row
    nrows = indptr.shape[0] - 1  # substract one because of format

    # output variables
    top_indices = np.zeros((nrows, k), dtype=indices.dtype) - 1
    top_vals = np.zeros((nrows, k), dtype=data.dtype) * np.nan

    for i in nb.prange(nrows):
        start, stop = indptr[i], indptr[i + 1]
        top_k = np.argsort(data[start:stop])[::-1][:k]
        n_items = min(len(top_k), k)
        # assign
        top_indices[i, 0:n_items] = indices[start:stop][top_k]
        top_vals[i, 0:n_items] = data[start:stop][top_k]

    return top_indices, top_vals


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
    top_indices, top_vals, rows, cols = _top_k_sparse_data(data, indices, indptr, k)
    top_indices, top_vals = np.concatenate(top_indices), np.concatenate(top_vals)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    top_indices = scipy.sparse.csr_matrix(
        (top_indices, (rows, cols)), dtype=indices.dtype)
    top_vals = scipy.sparse.csr_matrix((top_vals, (rows, cols)), dtype=data.dtype)

    return top_indices, top_vals


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
    if scipy.sparse.issparse(X):
        X = X.tocsr()
        data, indices, indptr = X.data, X.indices, X.indptr
        top_indices, top_vals = _top_k_dense(data, indices, indptr, k)

    else:
        top_indices = np.argsort(X, 1)[:, -k:][:, ::-1]
        top_vals = np.take_along_axis(X, top_indices, 1)

    return top_indices, top_vals


def sparse_chunks(M, chunk_size):
    """
    This creates copy since sparse matrices don't have views
    """
    n, _ = M.shape
    for i in range(0, n, chunk_size):
        start, stop = i, min(i + chunk_size, n)
        yield (start, stop), M[start:stop]
    if stop < n:
        yield (stop, n), M[stop:n]


def set_threshold(X, threshold, sparse_matrix=scipy.sparse.csr_matrix, copy=False):
    """
    Threshold a given (possibly sparse matrix). This function
    will increase the sparsity of the matrix. If the input is not sparse
    it will default to numpy functionality.

    Arguments
    =========
    X : sparse_matrix or np.array
    threshold : float
    sparse_matrix : output sparse matrix type
    copy : whether to operate in place (only for sparse input)

    >>> import scipy.sparse
    >>> X = scipy.sparse.dok_matrix((10, 10))
    >>> X[0, 0] = 0.5
    >>> X[2, 4] = 0.25
    >>> X[7, 1] = 0.75
    >>> X[8, 7] = 0.15
    >>> X = X.tocsr()
    >>> set_threshold(X, 0.1, copy=True).nnz
    4
    >>> set_threshold(X, 0.3, copy=True).nnz
    2
    >>> set_threshold(X, 0.8, copy=True).nnz
    0
    >>> _ = set_threshold(X, 0.5)
    >>> X.nnz
    2
    >>> X_orig = X.copy()
    >>> _ = set_threshold(X, 0.6, copy=True)
    >>> (X != X_orig).nnz
    0
    >>> _ = set_threshold(X, 0.6, copy=False)
    >>> (X != X_orig).nnz > 0
    True
    """
    if threshold == 0:
        return X

    if not scipy.sparse.issparse(X):
        if copy:
            X = np.copy(X)
        X[np.where(X < threshold)] = 0.0
        return X

    if copy:
        rows, cols, _ = scipy.sparse.find(X >= threshold)
        if len(rows) > 0:
            # matrix is not empty
            data = np.squeeze(np.array(X[rows, cols]), axis=0)  # find returns matrix
            return sparse_matrix(
                scipy.sparse.csr_matrix(
                    (data, (rows, cols)), shape=X.shape))
        else:
            # matrix is empty
            return sparse_matrix(X.shape)


    if isinstance(X, (scipy.sparse.lil_matrix, scipy.sparse.dok_matrix)):
        raise ValueError("Cannot efficiently drop items on", str(type(X)))

    X.data[np.abs(X.data) < threshold] = 0.0
    X.eliminate_zeros()

    return X
