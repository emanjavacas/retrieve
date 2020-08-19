
import math
import logging
import multiprocessing as mp

from scipy.sparse import dok_matrix
from tqdm import tqdm

from .search_index import SearchIndex

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

    def __call__(self, i):
        result = self.search_index.query(self.queries[i])
        return i, result


def parallel_search(search_index, queries, processes):
    workload = Workload(search_index, queries)

    sims = {}
    with mp.Pool(processes) as pool:
        for i, candidates in pool.map(workload, list(range(len(queries)))):
            for j, sim in candidates:
                sims[i, j] = sim

    return sims


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

        sims = dok_matrix((len(queries), len(index)))

        processes = mp.cpu_count() if processes < 0 else processes
        logger.info("Using {} cpus".format(processes))
        if processes == 1:
            for idx in tqdm(range(len(queries)), total=len(queries),
                            desc="Set similarity: {}".format(self.similarity_fn)):
                for jdx, sim in self.search_index.query(queries[idx]):
                    sims[idx, jdx] = sim
        else:
            results = parallel_search(self.search_index, queries, processes)
            for (i, j), sim in results.items():
                sims[i, j] = sim

        # drop self-similarities
        if self_search:
            sims.setdiag(0)

        # transform to csr
        sims = sims.tocsr()
        sims.eliminate_zeros()

        return sims
