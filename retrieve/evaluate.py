
import json
import contextlib
import collections

import numpy as np

from retrieve import sparse_utils


def _get_matches_fneg_tpos_at(q_idxs, i_idxs, ranking, at, strict):
    matches = []
    # [[i, i, i, ... (len(q_idxs))],
    #  [j, j, j, ...], ...
    #  [k, k, k, ...]]
    i_idxs_array = np.repeat(i_idxs[:, None], len(q_idxs), 1)
    # [i, j, k, i, j, k, ... len(q_idxs) * len(i_idxs)]
    i_idxs_array = np.ravel(i_idxs_array.T)
    # [[6, 4, 9, ... at],
    #   .
    #   .
    #   len(i_idxs)
    #  [3, 1, -1, -1, ... at],
    #   .
    #   .
    #   len(i_idxs)
    #  .
    #  .
    #  len(q_idxs) * len(i_idxs)
    q_idxs_array = np.repeat(ranking[q_idxs, :at], len(i_idxs), axis=0)

    row_matches, row_ranks = np.where(i_idxs_array[:, None] == q_idxs_array)
    # x-axis gives matches:
    # - rows of q_idxs (mod len i_idxs)
    # - rows of i_idxs (div len i_idxs)
    q_matches, i_matches = divmod(row_matches, len(i_idxs))
    # get original row and col indices
    q_matches, i_matches = q_idxs[q_matches], i_idxs[i_matches]
    # collect matches
    for q_match, i_match, rank in zip(q_matches.tolist(),
                                      i_matches.tolist(),
                                      row_ranks.tolist()):
        # rank should be 1-index
        matches.append((q_match, i_match, rank + 1))

    if strict:
        # how many were strictly retrieved
        tpos = len(q_matches)
        # how many of the many-to-many links were not retrieved
        fneg = len(q_idxs) * len(i_idxs) - len(q_matches)
    else:
        # at least one was retrieved
        # consider the whole match holistically
        tpos = int(len(matches) != 0)
        fneg = int(len(matches) == 0)

    return matches, tpos, fneg


def _get_fpos_at(unchecked, ranking, at):
    # empty ranks are represented as -1
    # considers a false positive if at least one indexed doc is retrieved within
    # the first ranked `at` documents
    fpos = np.sum(np.any(0 < ranking[unchecked, :at], axis=1))
    return fpos


def _get_tneg_at(ranking, refs, at):
    pred = np.sum(ranking[:, :at] == -1, axis=1) == at
    idxs = np.arange(len(ranking))
    true = np.array(list(set([i for tup, _  in refs for i in tup])))
    true = idxs[np.in1d(idxs, true, invert=True)]
    return np.sum(np.in1d(true, pred))


def ranking_stats_from_tuples(ranking, refs, at_values=(1, 5, 10, 20), strict=False):
    """
    Computes evaluation stats (true positives, false negatives and false positives)
    from a ranking matrix and an input set of true references.

    It considers a hit any link ranked within the first `at` items. Multiple
    cutoffs can be passed. The output is a list of dictionaries holding the results
    for the corresponding cutoff value.

    Arguments
    =========

    ranking : np.array(n, top_k), ranking array which has been pruned to the top k
        document hits
    refs : list[queries, indexed] where `queries` and `indexed` are tuples of doc_ids
        mapping to query or indexed documents that refer or are referred to by the
        other documents.
    at_values : tuple of cutoff points to consider for the evaluation. The intuition
        is that we can modify what we consider a hit based on how many of the top
        ranked items we are willing to check.
    strict : book, if False a reference is consider a true positive if at least one of
        the possibly many links in the reference is retrieved

    Output
    ======
    stats : list of dictionaries collecting the statistics, each dictionary is
        structured as follows: {'tpos': ..., 'fpos': ..., 'fneg': ..., 'tneg': ...}

    matches : list of tuples, where each tuple contains a match (q_id, i_d, rank)
        indicating which query doc was correctly retrieved as a reference to which
        indexed doc and at which rank

    >>> import numpy as np
    >>> n_queries, n_index = 5, 6
    >>> ranking = np.zeros((n_queries, n_index)) - 1
    >>> for i in range(n_queries):
    ...     for j in range(n_index):
    ...         if j < i:
    ...             ranking[i, j] = i + j
    >>> ranking
    array([[-1., -1., -1., -1., -1., -1.],
           [ 1., -1., -1., -1., -1., -1.],
           [ 2.,  3., -1., -1., -1., -1.],
           [ 3.,  4.,  5., -1., -1., -1.],
           [ 4.,  5.,  6.,  7., -1., -1.]])
    >>> refs = [([0], [1, 2]), ([2, 3], [3, 5, 6]), ([4], [8])]
    >>> stats, matches = ranking_stats_from_tuples(ranking, refs, at_values=[5])
    >>> len(stats)
    1
    >>> len(matches)
    3
    >>> matches
    [(2, 3, 2), (3, 3, 1), (3, 5, 3)]
    >>> stats = stats[0]
    >>> stats['tpos'], stats['fneg'], stats['fpos'], stats['tneg']
    (1, 2, 2, 1)
    >>> stats, matches = ranking_stats_from_tuples(
    ...     ranking, refs, at_values=[5], strict=True)
    >>> stats = stats[0]
    >>> stats['tpos'], stats['fneg'], stats['fpos'], stats['tneg']
    (3, 6, 2, 1)
    """
    # >>> refs = [([0], [1, 2]), ([2, 3], [3, 5, 6]), ([4], [8])]
    # >>> array([[-1,  -1,  -1,  -1,  -1, -1],  # fneg
    #            [ 1., -1,  -1,  -1,  -1, -1],  # fpos
    #            [ 2.,  3., -1,  -1,  -1, -1],  # tpos
    #            [ 3.,  4.,  5., -1,  -1, -1],  # (tpos already checked)
    #            [ 4.,  5.,  6.,  7., -1, -1]]) # fneg & fpos
    matches = []
    stats = [collections.Counter() for _ in at_values]
    checked = collections.defaultdict(list)

    for q_idxs, i_idxs in refs:
        q_idxs, i_idxs = np.array(q_idxs), np.array(i_idxs)
        for idx_at, at in enumerate(sorted(at_values)):
            row_matches, tpos, fneg = _get_matches_fneg_tpos_at(
                q_idxs, i_idxs, ranking, at, strict)
            stats[idx_at]['tpos'] += tpos
            stats[idx_at]['fneg'] += fneg

            # collect rank data only at the highest at
            # the rank can be used to filter out by cutoff points later
            if at == max(at_values):
                matches.extend(row_matches)

            if row_matches:
                row_checked, _, _ = zip(*row_matches)
                checked[at].extend(row_checked)

    # false positives and true negatives
    for idx, at in enumerate(sorted(at_values)):
        unchecked = np.arange(len(ranking))
        unchecked = unchecked[np.isin(unchecked, np.array(checked[at]), invert=True)]
        stats[idx]['fpos'] += _get_fpos_at(unchecked, ranking, at)
        stats[idx]['tneg'] += _get_tneg_at(ranking, refs, at)

    return stats, matches


def f_score(p, r, beta=1):
    """
    F-score
    """
    return (1 + beta ** 2) * ((p * r) / (((beta ** 2) * p) + r))


def get_metrics_from_tpos_fneg_fpos(tpos, fneg, fpos):
    p = tpos / (tpos + fpos)
    r = tpos / (tpos + fneg)
    return {'p': p,
            'r': r,
            'f1': f_score(p, r, beta=1),
            'f2': f_score(p, r, beta=2),
            'fp5': f_score(p, r, beta=0.5)}


def get_metrics(inp, refs, at_values=(1, 5, 10, 20), strict=False, input_type='sims'):
    """
    Get metrics from similarity metric and refs. Input might be the full similarity
    matrix or an already computed top-k ranking
    """
    assert inp.shape[1] >= max(at_values), \
        "Asked to rank top-{} items but got only {} items".format(
            max(at_values), inp.shape[1])

    ranking = None
    if input_type == 'sims':
        ranking, _ = sparse_utils.top_k(inp, max(at_values))
    else:
        ranking = inp[:, :max(at_values)]

    stats, _ = ranking_stats_from_tuples(
        ranking, refs, at_values=at_values, strict=strict)

    results = {}
    for at, stats in zip(at_values, stats):
        tpos, fneg, fpos = stats['tpos'], stats['fneg'], stats['fpos']
        # accumulate
        for metric, score in get_metrics_from_tpos_fneg_fpos(tpos, fneg, fpos).items():
            results['{}@{}'.format(metric, at)] = score

    return results


def get_thresholded_metrics_approx(sims, refs, thresholds, copy=False, **kwargs):
    # drop items that could never make it to the top k (add a safety margin)
    top_k = max(kwargs.get('at_values', [20]))
    max_items = min(sims.nnz, sims.shape[0] * top_k * 5)
    perc = 1 - (max_items / len(sims.data))
    min_th = np.percentile(sims.data, 100 * perc, interpolation='lower')
    sims = sparse_utils.set_threshold(sims, min_th, copy=True)
    assert len(sims.data) >= max_items

    return get_thresholded_metrics(sims, refs, thresholds, copy=copy, **kwargs)


def get_thresholded_metrics(sims, refs, thresholds, copy=False, **kwargs):
    thresholds = sorted(thresholds)
    metrics = collections.defaultdict(list)

    # sort sims
    ranking, values = sparse_utils.top_k(sims, max(kwargs.get('at_values', [20])))
    # overwrite input type
    kwargs['input_type'] = 'ranking'

    for th in thresholds:
        x, y = np.where(values < th)
        ranking[x, y] = -1
        for metric, val in get_metrics(ranking, refs, **kwargs).items():
            metrics[metric].append(val)

    return metrics





@contextlib.contextmanager
def results_writer(fp):
    """
    Write output of experiment to file. For a given similarity matrix and refs
    compute evaluation results at given thresholds. It also registers the sparsity
    of the passed matrix (nnz) *before* applying a cutoff at a top-k value.
    If copy==True, do not modify insplace the similarity matrix
    """
    fp = open(fp, 'w+')

    def write(sims, refs, thresholds, copy=False, **kwargs):
        row = {}
        row['results'] = []
        row['nnz'] = []

        metrics = get_thresholded_metrics(sims, refs, thresholds, copy=copy)

        for th, nnz in zip(thresholds, metrics.pop('nnz')):
            row['nnz'].append({'th': float(th), 'value': nnz})
        for metric, vals in metrics.items():
            for th, val in zip(thresholds, vals):
                row['results'].append({'metric': metric, 'th': float(th), 'val': val})
        for key, val in kwargs.items():
            row[key] = val

        fp.write(json.dumps(row) + '\n')
        fp.flush()

    yield write

    fp.close()
