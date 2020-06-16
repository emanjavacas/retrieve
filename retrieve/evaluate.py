
import collections

import numpy as np

from retrieve import sparse_utils


def _get_matches_fneg_tpos_at(q_idxs, i_idxs, ranking, at, strict):
    matches = []
    # [[i, i, i, ... (len(q_idxs))],
    #  [j, j, j, ...], ...
    i_idxs_array = np.repeat(i_idxs[:, None], len(q_idxs), 1)
    # [i, j, k, i, j, k, ... len(q_idxs)]
    i_idxs_array = np.ravel(i_idxs_array.T)
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
        rank += 1
        matches.append((q_match, i_match, rank))

    if strict:
        tpos = len(q_matches)
        fneg = len(q_idxs) * len(i_idxs) - len(q_matches)
    else:
        tpos = int(len(matches) != 0)
        fneg = int(len(matches) == 0)

    return matches, tpos, fneg


def _get_fpos_at(unchecked, ranking, at):
    # empty ranks are represented as -1
    # considers a false positive if at least one indexed doc is retrieved within
    # the first ranked `at` documents
    fpos = np.sum(np.any(0 < ranking[unchecked, :at], axis=1))
    return fpos


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
        structured as follows: {'tpos': ..., 'fpos': ..., 'fneg': ...}

    matches : list of tuples, where each tuple contains a match (q_id, i_d, rank)
        indicating which query doc was correctly retrieved as a reference to which
        indexed doc and at which rank

    >>> import numpy as np
    >>> n_queries, n_index = 5, 6
    >>> ranking = np.zeros((n_queries, n_index)) * np.nan
    >>> for i in range(n_queries):
    ...     for j in range(n_index):
    ...         if j < i:
    ...             ranking[i, j] = i + j
    >>> ranking
    array([[nan, nan, nan, nan, nan, nan],
           [ 1., nan, nan, nan, nan, nan],
           [ 2.,  3., nan, nan, nan, nan],
           [ 3.,  4.,  5., nan, nan, nan],
           [ 4.,  5.,  6.,  7., nan, nan]])
    >>> refs = [([0], [1, 2]), ([2, 3], [3, 5, 6]), ([4], [8])]
    >>> stats, matches = ranking_stats_from_tuples(ranking, refs, at_values=[5])
    >>> len(stats)
    1
    >>> len(matches)
    3
    >>> matches
    [(2, 3, 2), (3, 3, 1), (3, 5, 3)]
    >>> stats = stats[0]
    >>> stats['tpos'], stats['fneg'], stats['fpos']
    (1, 2, 0)
    >>> stats, matches = ranking_stats_from_tuples(
    ...     ranking, refs, at_values=[5], strict=True)
    >>> stats = stats[0]
    >>> stats['tpos'], stats['fneg'], stats['fpos']
    (3, 6, 0)
    """
    matches, checked = [], []
    stats = [collections.Counter() for _ in at_values]

    for q_idxs, i_idxs in refs:
        checked.extend(q_idxs)
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

    # false positives
    checked = np.array(list(set(checked)))  # avoid duplicates
    unchecked = np.arange(len(ranking))
    unchecked = unchecked[np.isin(unchecked, checked, invert=True)]
    for idx, at in enumerate(sorted(at_values)):
        stats[idx]['fpos'] += _get_fpos_at(unchecked, ranking, at)

    return stats, matches


def f_score(p, r, beta=1):
    """
    F-score
    """
    return (1 + beta ** 2) * ((p * r) / (((beta ** 2) * p) + r))


def get_metrics(sims, refs, at_values=(1, 5, 10, 20), strict=False):
    """
    Get metrics from similarity metric and refs
    """
    ranking, sims = sparse_utils.top_k(sims, max(at_values))
    stats, matches = ranking_stats_from_tuples(
        ranking, refs, at_values=at_values, strict=strict)

    results = {}
    for at, stats in zip(at_values, stats):
        tpos, fneg, fpos = stats['tpos'], stats['fneg'], stats['fpos']
        # precision
        p = tpos / (tpos + fpos)
        # recall
        r = tpos / (tpos + fneg)
        # f-score (0.5, 1, 2)
        f1 = f_score(p, r, beta=1)
        f2 = f_score(p, r, beta=2)
        fp5 = f_score(p, r, beta=0.5)

        # accumulate
        for score, metric in zip([p, r, fp5, f1, f2], ['p', 'r', 'f0.5', 'f1', 'f2']):
            results['{}@{}'.format(metric, at)] = score

    return results


# import numpy as np
# n_queries, n_index = 5, 6
# ranking = np.zeros((n_queries, n_index)) * np.nan
# for i in range(n_queries):
#     for j in range(n_index):
#         if j < i:
#             ranking[i, j] = i + j
# refs = [([0], [1, 2]), ([2, 3], [3, 5, 6]), ([4], [8])]
# stats, matches = ranking_stats_from_tuples(ranking, refs, at_values=[5], strict=True)