
from retrieve.compare import SearchIndex

from scipy.sparse import dok_matrix
from tqdm import tqdm


def containment(s1, s2, **kwargs):
    s1, s2 = set(s1), set(s2)
    return len(s1 & s2) / (len(s1) or 1)


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


class get_similarities:
    def __init__(self, search_index, queries):
        self.search_index = search_index
        self.queries = queries

    def __call__(self, i):
        return i, self.search_index.query(self.queries[i])


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

    def get_similarities(self, queries, index=None):
        self_search = index is None
        if index is None:
            index = queries

        self.fit(index, queries=queries)

        S = dok_matrix((len(queries), len(index)))

        # do the search
        for idx in tqdm(range(len(queries)), total=len(queries),
                        desc="Running {}".format(self.similarity_fn)):
            for jdx, sim in self.search_index.query(queries[idx]):
                S[idx, jdx] = sim

        # drop self-similarities
        if self_search:
            S.setdiag(0)

        # transform to csr
        S = S.tocsr()
        S.eliminate_zeros()

        return S
