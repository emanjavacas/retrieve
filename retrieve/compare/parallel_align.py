
import multiprocessing as mp
import numpy as np
import scipy.sparse

from retrieve.compare.align import local_alignment


class Workload:
    def __init__(self, coll1, coll2, field='lemma', use_features=False, **kwargs):
        self.coll1 = coll1
        self.coll2 = coll2
        self.field = field
        self.use_features = use_features
        self.kwargs = kwargs

    def __call__(self, tup):
        # unpack
        i, j = tup
        if self.use_features:
            s1, s2 = self.coll1[i].features, self.coll2[j].features
        else:
            s1, s2 = self.coll1[i].fields[self.field], self.coll2[j].fields[self.field]

        return (i, j), local_alignment(s1, s2, **self.kwargs)


def align_collections(queries, index=None, field='lemma', S=None, processes=-1, **kwargs):

    if index is None:
        index = queries

    # get target ids
    if S is not None:
        x, y, _ = scipy.sparse.find(S)
    else:
        x, y = np.meshgrid(np.arange(len(queries)), np.arange(len(index)))
        x, y = x.reshape(-1), y.reshape(-1)

    x, y = x.tolist(), y.tolist()

    workload = Workload(queries, index, field, **kwargs)

    sims = scipy.sparse.dok_matrix((len(queries), len(index)))  # sparse output
    processes = processes if processes > 0 else mp.cpu_count()
    with mp.Pool(processes) as pool:
        for (i, j), (*_, score) in pool.map(workload, zip(x, y)):
            sims[i, j] = score

    return sims


if __name__ == '__main__':
    import timeit

    from retrieve.corpora import load_vulgate
    from retrieve.data import Criterion, TextPreprocessor, FeatureSelector
    from retrieve import utils
    from retrieve.compare.align import create_embedding_scorer
    from retrieve.set_similarity import SetSimilarity

    # load
    vulg = load_vulgate(max_verses=1000)
    # preprocess
    TextPreprocessor().process_collections(vulg, min_n=2, max_n=4)
    # drop features and get vocabulary
    FeatureSelector(vulg).filter_collections(
        vulg, (Criterion.DF >= 2) & (Criterion.FREQ >= 5))
    # get documents
    feats = vulg.get_features(cast=set)
    # set-based similarity
    S = SetSimilarity(0.5, similarity_fn="containment").get_similarities(feats)

    # alignment
    TextPreprocessor().process_collections(vulg)
    vocab = FeatureSelector(vulg).filter_collections(
        vulg, (Criterion.DF >= 2) & (Criterion.FREQ >= 5))
    # load embeddings, make sure S is in same order as vocab
    embs = utils.Embeddings.from_csv('latin.lemma.embeddings', vocab=vocab)
    # embedding scorer
    scorer = create_embedding_scorer(embs)

    x, y, _ = scipy.sparse.find(S)
    print("Considering {} comparisons".format(len(x)))
    time = timeit.Timer(lambda: align_collections(vulg, vulg, S=S)).timeit(5)
    print(" - Took", time)
