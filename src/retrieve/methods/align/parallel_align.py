
import logging
import multiprocessing as mp

import numpy as np
import scipy.sparse
import tqdm


logger = logging.getLogger(__name__)


class Workload:
    def __init__(self, coll1, coll2, field='lemma', **kwargs):
        self.coll1 = coll1
        self.coll2 = coll2
        self.field = field
        self.kwargs = kwargs

    def __call__(self, args):
        # unpack
        i, j = args
        score = self.coll1[i].local_alignment(
            self.coll2[j], field=self.field, only_score=True, **self.kwargs)

        return (i, j), score


def align_collections(queries, index=None, S=None, field=None, processes=1, **kwargs):

    if index is None:
        index = queries

    # get target ids
    if S is not None:
        x, y, _ = scipy.sparse.find(S)
    else:
        x, y = np.meshgrid(np.arange(len(queries)), np.arange(len(index)))
        x, y = x.reshape(-1), y.reshape(-1)

    x, y = x.tolist(), y.tolist()

    sims = scipy.sparse.dok_matrix((len(queries), len(index)))  # sparse output

    processes = mp.cpu_count() if processes < 0 else processes
    logger.info("Using {} CPUs".format(processes))
    if processes == 1:
        for i, j in tqdm.tqdm(zip(x, y), total=len(x), desc='Local alignment'):
            score = queries[i].local_alignment(
                index[j], field=field, only_score=True, **kwargs)
            sims[i, j] = score
    else:
        workload = Workload(queries, index, field=field, **kwargs)
        with mp.Pool(processes) as pool:
            for (i, j), score in pool.map(workload, list(zip(x, y))):
                sims[i, j] = score

    return sims.tocsr()


if __name__ == '__main__':
    import timeit

    from retrieve.corpora import load_vulgate
    from retrieve.data import Criterion, TextPreprocessor, FeatureSelector
    from retrieve.embeddings import Embeddings
    from retrieve.methods import create_embedding_scorer, SetSimilarity

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
    embs = Embeddings.from_resource(
        'latin.lemma.ft.dim100.mc2.embeddings.gz', vocab=vocab)
    # embedding scorer
    scorer = create_embedding_scorer(embs)

    x, y, _ = scipy.sparse.find(S)
    print("Considering {} comparisons".format(len(x)))
    time = timeit.Timer(lambda: align_collections(vulg, vulg, S=S)).timeit(5)
    print(" - Took", time)
