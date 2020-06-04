
import multiprocessing as mp
import numpy as np
import scipy.sparse

from . import align


class Workload:
    def __init__(self, coll1, coll2, field='lemma', **kwargs):
        self.coll1 = coll1
        self.coll2 = coll2
        self.field = field
        self.kwargs = kwargs

    def __call__(self, tup):
        # unpack
        i, j = tup

        return (i, j), align.local_alignment(
            self.coll1[i].fields[self.field],
            self.coll2[j].fields[self.field],
            **self.kwargs)


def align_collections(coll1, coll2, field='lemma', S=None, processes=-1, **kwargs):
    output = {}  # sparse output

    # get target ids
    if S is not None:
        x, y, _ = scipy.sparse.find(S)
    else:
        x, y = np.meshgrid(np.arange(len(coll1)), np.arange(len(coll2)))
        x, y = x.reshape(-1), y.reshape(-1)

    workload = Workload(coll1, coll2, field, **kwargs)

    processes = processes if processes > 0 else mp.cpu_count()
    with mp.Pool(processes) as pool:
        for (i, j), data in pool.map(workload, zip(x, y)):
            output[i, j] = data

    return output
