
import logging
import collections

import numpy as np
from numpy.lib.function_base import _sinc_dispatcher
import scipy.sparse as sparse

from retrieve import utils, sparse_utils
from retrieve.data import Criterion, TextPreprocessor, FeatureSelector
from retrieve.embeddings import Embeddings
from retrieve.methods import SetSimilarity, Tfidf, align_collections
from retrieve.methods import create_embedding_scorer, ConstantScorer


logger = logging.getLogger(__name__)


def require_embeddings(embs, msg='', **kwargs):
    if isinstance(embs, str):
        embs = Embeddings.from_file(embs, **kwargs)
    if not isinstance(embs, Embeddings):
        raise ValueError(msg)
    return embs


def get_vocab_from_colls(*colls, field=None):
    output = collections.Counter()
    for coll in colls:
        for doc in coll:
            output.update(doc.get_features(field=field))
    output, _ = zip(*output.most_common())
    return output


class Match:
    def __init__(self, doc1, doc2, sim):
        self.doc1 = doc1
        self.doc2 = doc2
        self.sim = sim

    def __repr__(self):
        return 'Similarity -> {:.5f}\n\t{}: {}\n\t{}: {}'.format(
            self.sim,
            self.doc1.doc_id, self.doc1.text,
            self.doc2.doc_id, self.doc2.text)


class Results:
    def __init__(self, sims, coll1, coll2):
        self.sims = sparse.csr_matrix(sims)
        self.coll1 = coll1
        self.coll2 = coll2

    @property
    def nnz(self):
        return self.sims.nnz

    def keep_top_k(self, k):
        if self.nnz < k:
            logger.info("Nothing to drop from similarity matrix")
            return

        k_val = np.argsort(self.sims)[-k]
        sparse_utils.set_threshold(self.sims, k_val)

    def drop_sims(self, min_sim):
        sparse_utils.set_threshold(self.sims, min_sim)

    def get_top_matches(self, n=None, min_sim=0, max_sim=None, sample=False):
        x, y, _ = sparse.find(self.sims > min_sim)
        score = self.sims[x, y]
        # sparse.find returns a scipy matrix instead of a np array
        score = np.array(score)[0]
        if max_sim is not None:
            index, = np.where(score <= max_sim)
            x, y, score = x[index], y[index], score[index]
        index = np.argsort(score)[::-1]
        if sample and n is not None:
            index = np.random.choice(np.arange(len(index)), size=n, replace=False)
        n = n or len(score)
        for i in index[:n]:
            yield Match(self.coll1[x[i]], self.coll2[y[i]], score[i])


def pipeline(coll1, coll2=None,
             # Text Preprocessing
             field='lemma', lower=True, stopwords=None, stop_field='lemma',
             drop_punctuation=True, punct_field='token',
             # Ngrams
             min_n=1, max_n=1, skip_k=0, sep='--',
             # Feature Selection
             criterion=None,
             # Method params
             method='set-based', threshold=0, processes=-1, embs=None, chunk_size=5000,
             # Set-based
             # - SetSimilarity: similarity_fn
             #     ('containment', 'containment_min', 'jaccard')
             # VSM-based
             # - Tfidf: vocab, **sklearn,feature_extraction.text.TfidfVectorizer
             # Alignment-based
             # - match, mismatch, open_gap, extend_gap, cutoff, beta
             method_params={},
             # Soft_cosine_params: cutoff, beta
             use_soft_cosine=False, soft_cosine_params={},
             # whether to use parallel soft-cosine (could run into memory issues)
             parallel_soft_cosine=False,
             # For Blast-style alignment
             precomputed_sims=None,
             # return time stats
             return_stats=False, verbose=False):

    """
    Provides an interface to a full, highly configurable text reuse pipeline
    
    Arguments
    =========
    coll1 : retrieve.data.Collection, target collection
    coll2 : retrieve.data.Collection (optional), source collection, if not given
        the pipeline will comput intra-collection reuse
    
    Text preprocessing arguments
    -----------------------------
    field : str (default='lemma'), field to use as features
    lower : bool (default=True), whether to lowercase all input features
    stopwords : retrieve.utils.Stopwords (optional), stopwords to filter out
    stop_field : str (default='lemma'), field to use for stopwords
    drop_punctuation : bool (default=True), whether to remove punctuation
    punct_field : str, (default='token'), field to use for punctuation
    min_n : int (default=1), minimum token-level n-gram size
    max_n : int (default=1), maximum token-level n-gram size
    skip_k : int, (default=0), skip-grams skipping every `skip_k` tokens
    sep : str (default='--'), string to use for binding n-grams
    
    Feature Selection arguments
    ---------------------------
    criterion : retrieve.data.Criterion (optional), 

    Method arguments
    ----------------
    method : str (default='set-based'), method to use. One of
        "set-based", "vsm-based" or "alignment-based".
    method_params : dict, parameters passed to the method. These change
        depending on the method used.
        * set-based
            - similarity_fn : str, one of ('containment', 'containment_min', 'jaccard')
        * vsm-based
            - parameters passed to the **sklearn,feature_extraction.text.TfidfVectorizer
        * alignment-based
            - scorer : a retrieve.methods.BaseScorer that computes the scores for the
                input sequences
            - open_gap : float, penalty for opening a gap
            - extend_gap : float, penalty for extending a gap
            - cutoff : float (between 0 and 1), only used if `use_soft_cosine` is True.
                Ignore word-to-word similarities below this threshold
            - beta : float, only used if `use_soft_cosine` is True. Exponent of
                word-to-word cosine similarities. This helps skewing or flattening the
                distribution of similarities
    use_soft_cosine : bool, (default=False), whether to use soft-cosine (requires `embs`).
        Only used if `method` is 'vsm-based'
    soft_cosine_params : dict, parameters for the soft cosine method. Only used if
        `use_soft_cosine` is passed and `method` is 'vsm-based'.
        - cutoff : float, only used if passing `embs` (same as in alignment-based)
        - beta : float, only used if passing `embs` (same as in alignment-based)
    parallel_soft_cosine : bool (default=True), whether to parallelize the computation of
        soft-cosine.
    threshold : float (default=0), minimum similarity to retain the match. This helps
        alleviating memory pressure, especially when processing large collections.
    processes : float (default=-1 meaning use all available cores), number of cores to use
        during the computation.
    embs : retrieve.utils.Embeddings, only needed if `method` is 'alignment-based' or
        'vsm-based' and `use_soft_cosine` is True.
    chunk_size : int (default=5000), number of instances to consider at a time from the 
        target collection (coll1). For very large collections, this may be necessary in
        in combination with an appropriate `threshold` in order to make the result fit
        in memory.
    """

    colls = [coll for coll in [coll1, coll2] if coll is not None]

    if isinstance(stopwords, str):
        stopwords = utils.Stopwords(stopwords)

    stats = {}

    with utils.timer() as timer:
        # * preprocessing
        TextPreprocessor(
            field=field, lower=lower, stopwords=stopwords, stop_field=stop_field,
            drop_punctuation=drop_punctuation, punct_field=punct_field,
        ).process_collections(
            *colls, min_n=min_n, max_n=max_n, skip_k=skip_k, sep=sep)
        fsel = FeatureSelector(*colls)
        # get selected vocabulary
        vocab = fsel.filter_collections(*colls, criterion=criterion)

        stats['preprocessing'] = timer(desc='Preprocessing')

        # * similarities
        # - set-based method
        if method.startswith('set'):
            coll1_feats = coll1.get_features(cast=set)
            coll2_feats = coll2.get_features(cast=set) if coll2 else coll1_feats
            sims = SetSimilarity(threshold, **method_params).get_similarities(
                coll1_feats, coll2_feats, processes=processes)

        # - vsm-based method
        elif method.startswith('vsm'):
            coll1_feats = coll1.get_features()
            coll2_feats = coll2.get_features() if coll2 is not None else coll1_feats
            tfidf = Tfidf(vocab, **method_params).fit(coll1_feats + coll2_feats)
            if use_soft_cosine:
                embs = require_embeddings(
                    embs, vocab=get_vocab_from_colls(coll1, coll2, field=field),
                    msg='soft cosine requires embeddings')
                sims = tfidf.get_soft_cosine_similarities(
                    coll1_feats, coll2_feats, embs=embs,
                    threshold=threshold,
                    chunk_size=chunk_size, parallel=parallel_soft_cosine,
                    **soft_cosine_params)
            else:
                sims = tfidf.get_similarities(
                    coll1_feats, coll2_feats, threshold=threshold)

        # - alignment-based
        elif method.startswith('alignment'):
            # get scorer
            if 'scorer' in method_params:
                scorer = method_params['scorer']
            elif embs is not None:
                vocab = get_vocab_from_colls(coll1, coll2, field=field)
                scorer = create_embedding_scorer(
                    require_embeddings(embs, vocab=vocab),
                    vocab=vocab,  # in case embeddings are already loaded
                    **{key: val for key, val in method_params.items()
                       if key in set(['match', 'mismatch', 'cutoff', 'beta'])})
            else:
                scorer = ConstantScorer(
                    **{key: val for key, val in method_params.items()
                       if key in set(['match', 'mismatch'])})

            if precomputed_sims is not None:
                logger.info("Computing {} alignments...".format(precomputed_sims.nnz))
            sims = align_collections(
                coll1, coll2,
                S=precomputed_sims, field=field, processes=processes, scorer=scorer,
                **{key: val for key, val in method_params.items()
                   if key in set(['extend_gap', 'open_gap'])})
        else:
            raise ValueError("Unknown method", method)

        stats['similarity'] = timer(desc='Similarity')

    if return_stats:
        return sims, stats

    return sims
