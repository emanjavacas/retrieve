
import multiprocessing as mp

import tqdm
import numpy as np
import scipy.sparse
from scipy.optimize import linear_sum_assignment

from .. import utils


def get_linear_sum_assignment_score(S, vocab1, vocab2, doc1, doc2, normalize=True):
    scores = np.zeros((len(doc1), len(doc2)))
    for idx, w1 in enumerate(doc1):
        for jdx, w2 in enumerate(doc2):
            scores[idx, jdx] = S[vocab1[w1], vocab2[w2] + len(vocab1)]

    a, b = linear_sum_assignment(scores, maximize=True)
    score = scores[a, b].sum()

    if normalize:
        score /= len(doc2)

    return score


class Workload:
    def __init__(self, S, vocab1, vocab2, coll1, coll2):
        self.S = S
        self.vocab1 = vocab1
        self.vocab2 = vocab2
        self.coll1 = coll1
        self.coll2 = coll2

    def __call__(self, chunk):
        output = []
        for i, j in chunk:
            doc1, doc2 = self.coll1[i].get_features(), self.coll2[j].get_features()
            score = get_linear_sum_assignment_score(
                self.S, self.vocab1, self.vocab2, self.coll1, self.coll2, self.normalize)
            output.append((i, j, score))
        return output


def parallel_search(S, vocab1, vocab2, coll1, coll2, items, processes, **kwargs):
    workload = Workload(S, vocab1, vocab2, coll1, coll2)
    rows, cols, data = [], [], []

    with mp.Pool(processes) as pool:
        chunks = list(utils.chunks(items, len(items) // (processes + 1)))
        for chunk in enumerate(pool.map(workload, chunks)):
            for i, j, score in chunk:
                rows.append(i)
                cols.append(j)
                data.append(score)

    return rows, cols, data


def parallel_linear_sum_assignment(
        coll1, coll2, S, vocab1, vocab2, pre_sims, normalize=True, processes=1):
    x, y, _ = scipy.sparse.find(pre_sims)
    
    processes = mp.cpu_count() if processes < 0 else processes
    if processes == 1:
        sims = scipy.sparse.dok_matrix((len(coll1), len(coll2)))
        for i, j in tqdm.tqdm(zip(x, y), total=len(x), desc='Linear Sum Assignment'):
            score = get_linear_sum_assignment_score(
                S, vocab1, vocab2, coll1[i].get_features(), coll2[j].get_features(),
                normalize=normalize)

    else:
        items = list(zip(x, y))
        rows, cols, data = parallel_search(
            S, vocab1, vocab2, coll1, coll2, items, processes, normalize=normalize)
        sims = scipy.sparse.csr_matrix(
            (data, (rows, cols)), shape=(len(coll1), len(coll2)))

    sims = sims.tocsr()
    sims.eliminate_zeros()

    return sims
