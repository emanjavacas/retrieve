
import multiprocessing as mp
import numpy as np
import scipy.sparse

from retrieve.compare.align import local_alignment


class Workload:
    def __init__(self, coll1, coll2, field='lemma', **kwargs):
        self.coll1 = coll1
        self.coll2 = coll2
        self.field = field
        self.kwargs = kwargs

    def __call__(self, tup):
        # unpack
        i, j = tup

        return (i, j), local_alignment(
            self.coll1[i].fields[self.field],
            self.coll2[j].fields[self.field],
            **self.kwargs)


def align_collections(coll1, coll2=None,
                      field='lemma',
                      S=None, doc_ids=None,
                      processes=-1, **kwargs):

    if coll2 is None:
        coll2 = coll1

    # get target ids
    if S is not None:
        x, y, _ = scipy.sparse.find(S)
    else:
        x, y = np.meshgrid(np.arange(len(coll1)), np.arange(len(coll2)))
        x, y = x.reshape(-1), y.reshape(-1)

    if doc_ids is not None:
        doc_ids = np.array(doc_ids)
        x, y = doc_ids[x], doc_ids[y]

    x, y = x.tolist(), y.tolist()

    workload = Workload(coll1, coll2, field, **kwargs)

    sims = scipy.sparse.dok_matrix((len(coll1), len(coll2)))  # sparse output
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
    vulg = load_vulgate(include_blb=True, max_verses=1000)
    # preprocess
    TextPreprocessor().process_collections(vulg, min_n=2, max_n=4)
    # drop features and get vocabulary
    FeatureSelector(vulg).filter_collections(
        vulg, (Criterion.DF >= 2) & (Criterion.FREQ >= 5))
    # get documents
    feats, doc_ids = vulg.get_nonempty_features(cast=set)
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
    print("Considering {}/{} comparisons".format(len(x), len(doc_ids) ** 2))
    time = timeit.Timer(
        lambda: align_collections(vulg, vulg, S=S, doc_ids=doc_ids)
    ).timeit(5)
    print(" - Took", time)
