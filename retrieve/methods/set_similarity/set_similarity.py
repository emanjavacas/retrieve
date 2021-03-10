
import math
import logging
import multiprocessing as mp

from scipy.sparse import dok_matrix, csr_matrix
from tqdm import tqdm

from .search_index import SearchIndex
from ... import utils

logger = logging.getLogger(__name__)


def cosine(s1, s2, **kwargs):
    s1, s2 = set(s1), set(s2)
    return len(s1 & s2) / (math.sqrt(len(s1)) * math.sqrt(len(s2)))


def containment(s1, s2, **kwargs):
    s1, s2 = set(s1), set(s2)
    return len(s1 & s2) / (len(s1) or 1)


def containment_min(s1, s2, **kwargs):
    s1, s2 = set(s1), set(s2)
    return len(s1 & s2) / max(len(s1), len(s2))


def jaccard(s1, s2, **kwargs):
    s1, s2 = set(s1), set(s2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def weighted_jaccard(s1, s2, **kwargs):
    den = num = 0
    for w in set(s1).union(s2):
        c1, c2 = s1.get(w, 0), s2.get(w, 0)
        num += min(c1, c2)
        den += max(c1, c2)
    return num / den


def weighted_containment(s1, s2, **kwargs):
    den = num = 0
    for w in set(s1).union(s2):
        c1, c2 = s1.get(w, 0), s2.get(w, 0)
        num += min(c1, c2)
        den += c2
    return num / den


class Workload:
    def __init__(self, search_index, queries):
        self.search_index = search_index
        self.queries = queries

    def __call__(self, chunk):
        output = []
        for i in chunk:
            result = self.search_index.query(self.queries[i])
            output.append((i, result))
        return output


def parallel_search(search_index, queries, processes):
    workload = Workload(search_index, queries)
    rows, cols, data = [], [], []

    with mp.Pool(processes) as pool:
        chunks = list(utils.chunks(
            range(len(queries)),
            # make sure to use at most `processes` chunks
            len(queries) // (processes + 1)))
        for idx, chunk in enumerate(pool.map(workload, chunks)):
            for i, results in chunk:
                for j, sim in results:
                    rows.append(i)
                    cols.append(j)
                    data.append(sim)

    return rows, cols, data


class SetSimilarity:
    """
    Approximate set similarity

    similarity_fn : one of ...
    """
    def __init__(self, threshold, similarity_fn='containment'):
        self.threshold = threshold
        self.similarity_fn = similarity_fn
        self.search_index = None

    def fit(self, index, queries=None):
        self.search_index = SearchIndex(
            index, queries=queries,
            similarity_func_name=self.similarity_fn,
            similarity_threshold=self.threshold)
        return self

    def get_similarities(self, queries, index=None, processes=1):
        self_search = index is None
        if index is None:
            index = queries

        self.fit(index, queries=queries)

        processes = mp.cpu_count() if processes < 0 else processes
        logger.info("Using {} CPUs".format(processes))
        if processes == 1:
            sims = dok_matrix((len(queries), len(index)))
            for idx in tqdm(range(len(queries)), total=len(queries),
                            desc="Set similarity: {}".format(self.similarity_fn)):
                for jdx, sim in self.search_index.query(queries[idx]):
                    sims[idx, jdx] = sim
        else:
            results = parallel_search(self.search_index, queries, processes)
            rows, cols, data = results
            sims = csr_matrix((data, (rows, cols)), shape=(len(queries), len(index)))

        # drop self-similarities
        if self_search:
            sims.setdiag(0)
        # transform to csr
        sims = sims.tocsr()
        sims.eliminate_zeros()

        return sims
