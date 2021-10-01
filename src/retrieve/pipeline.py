
import logging

import numpy as np
import scipy.sparse as sparse

from retrieve import utils, sparse_utils
from retrieve.data import TextPreprocessor, FeatureSelector, get_vocab_from_colls
from retrieve.embeddings import Embeddings
from retrieve.methods import SetSimilarity, Tfidf, align_collections
from retrieve import methods


logger = logging.getLogger(__name__)


class Match:
    def __init__(self, doc1, doc2, sim, 
                 with_context=False,
                 coll1=None, coll2=None, n_words=25):

        self.with_context = with_context
        if with_context:
            self._meta = Match.get_match_with_context(
                doc1, doc2, sim, coll1, coll2, n_words=n_words)
        else:
            self._meta = Match.get_match(doc1, doc2, sim)

        # add horizontal alignment
        try:
            a1, a2, _ = doc1.local_alignment(doc2, field='lemma')
        except:
            a1, a2, _ = doc1.local_alignment(doc2, field='token')
        self._horizontal_alignment = doc1.get_horizontal_alignment(
            doc2, a1=a1, a2=a2, field='token')

    def __repr__(self):
        return self.get_printable_match()

    def get_data(self):
        return self._meta

    def print_alignment(self):
        print('Similarity -> {:.5f}\n\t{}\n\t{}\n\t{}'.format(
                self._meta['similarity'], 
                self._meta['doc1id'], 
                '\t{}\n\t\t{}\n\t\t{}'.format(*self._horizontal_alignment),
                self._meta['doc2id']))

    @staticmethod
    def get_match(doc1, doc2, sim):
        return {
            'similarity': sim, 
            'doc1id': doc1.doc_id, 
            'doc1text': doc1.text,
            'doc2id': doc2.doc_id, 
            'doc2text': doc2.text}

    @staticmethod
    def get_match_with_context(doc1, doc2, sim, coll1, coll2, n_words=25):
        doc1left = coll1.get_doc_context_left(coll1.get_doc_idx(doc1.doc_id), n_words)
        doc1right = coll1.get_doc_context_right(coll1.get_doc_idx(doc1.doc_id), n_words)
        doc2left = coll2.get_doc_context_left(coll2.get_doc_idx(doc2.doc_id), n_words)
        doc2right = coll2.get_doc_context_right(coll2.get_doc_idx(doc2.doc_id), n_words)
        return {
            'similarity': sim, 
            'doc1id': doc1.doc_id,
            'doc1left': ' '.join(doc1left),
            'doc1text': doc1.text,
            'doc1right': ' '.join(doc1right),
            'doc2id': doc2.doc_id, 
            'doc2left': ' '.join(doc2left),
            'doc2text': doc2.text,
            'doc2right': ' '.join(doc2right)}

    def get_printable_match(self):
        if self.with_context:
            return 'Similarity -> {:.5f}\n\t{}:\n\t{}\n\n\t{}:\n\t{}'.format(
                self._meta['similarity'],
                self._meta['doc1id'],
                '{}\n\t    {}\n\t{}'.format(
                    self._meta['doc1left'], self._meta['doc1text'], self._meta['doc1right']),
                self._meta['doc2id'],
                '{}\n\t    {}\n\t{}'.format(
                    self._meta['doc2left'], self._meta['doc2text'], self._meta['doc2right']))
        else:
            return 'Similarity -> {:.5f}\n\t{}: {}\n\t{}: {}'.format(
                self._meta['similarity'], 
                self._meta['doc1id'], 
                self._meta['doc1text'],
                self._meta['doc2id'],
                self._meta['doc2text'])


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

    def get_top_matches(self, n=None, min_sim=0, max_sim=None, sample=False, 
                        with_context=False, n_words=25, filter_func=None,
                        whitelist_x=None, whitelist_y=None,
                        blacklist_x=None, blacklist_y=None):

        x, y, _ = sparse.find(self.sims > min_sim)
        score = self.sims[x, y]
        # sparse.find returns a scipy matrix instead of a np array
        score = np.array(score)[0]
        if max_sim is not None:
            keep, = np.where(score <= max_sim)
            x, y, score = x[keep], y[keep], score[keep]

        # white/blacklisting
        if whitelist_x is not None:
            drop, = np.where(~np.isin(x, whitelist_x))
            x, y, score = np.delete(x, drop), np.delete(y, drop), np.delete(score, drop)
        elif blacklist_x is not None:
            drop, = np.where(np.isin(x, blacklist_x))
            x, y, score = np.delete(x, drop), np.delete(y, drop), np.delete(score, drop)
        if whitelist_y is not None:
            drop, = np.where(~np.isin(y, whitelist_y))
            x, y, score = np.delete(x, drop), np.delete(y, drop), np.delete(score, drop)
        elif blacklist_y is not None:
            drop, = np.where(np.isin(y, blacklist_y))
            x, y, score = np.delete(x, drop), np.delete(y, drop), np.delete(score, drop)

        index = np.argsort(score)[::-1]
        if sample and n is not None:
            index = np.random.choice(np.arange(len(index)), size=n, replace=False)
        n, i = n or len(score), 0
        while i < n:
            doc1, doc2, sim = self.coll1[x[i]], self.coll2[y[i]], score[i]
            # filter out based on filter function
            if filter_func is not None and filter_func(doc1, doc2, sim):
                continue
            yield Match(
                doc1, doc2, sim, 
                with_context=with_context, 
                coll1=self.coll1, coll2=self.coll2, n_words=n_words)
            i += 1

    def export_top_matches_to_csv(self, outputpath, tuple_sep='+++', **kwargs):
        with open(outputpath, 'w+') as f:
            header = None
            for m in self.get_top_matches(**kwargs):
                if header is None:
                    header = list(m.get_data())
                    f.write('\t'.join(header) + '\n')
                row = ''
                for key, val in m.get_data().items():
                    if isinstance(val, float):
                        val = '{:.5f}'.format(val)
                    elif isinstance(val, tuple):
                        val = tuple_sep.join(map(str, val))
                    else:
                        val = str(val)
                    row += ('\t' if row else '') + val
                f.write(row + '\n')


def pipeline(coll1, coll2=None,
             # Text Preprocessing
             field='lemma', lower=True, stopwords=None, stop_field='lemma',
             drop_punctuation=True, punct_field='token', field_regexes={},
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
    field_regexes : dict(str, dict(str, regex)), A dictionary mapping fields to
        a dictionary with {'regex': regex, 'should_match': bool}, where `regex`
        is a regex applied on each token of the target field, and `should_match`
        specifies whether the regex is positive or negative.
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
            - scorer_class : str, "ConstantScorer", "EmbeddingScorer", "LookupScorer",
                only used if `scorer` is not passed. Defaults to "ConstantScorer"
            - scorer_params : dict, parameters passed the scorer class, only used if
                `scorer` is not passed
                + ConstantScorer
                    - match : float, reward for a match
                    - mismatch : float, penalty for a mismatch
                + EmbeddingScorer (additionally)
                    - cutoff : float (between 0 and 1), only used if `use_soft_cosine` is True.
                        Ignore word-to-word similarities below this threshold
                    - beta : float, only used if `use_soft_cosine` is True. Exponent of
                        word-to-word cosine similarities. This helps skewing or flattening the
                        distribution of similarities
            - open_gap : float, penalty for opening a gap
            - extend_gap : float, penalty for extending a gap
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
            field_regexes=field_regexes
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
                sims = tfidf.get_soft_cosine_similarities(
                    coll1_feats, coll2_feats, 
                    embs=Embeddings.require_embeddings(
                        embs, vocab=get_vocab_from_colls(coll1, coll2, field=field),
                        msg='soft cosine requires embeddings'),
                    threshold=threshold, chunk_size=chunk_size,
                    parallel=parallel_soft_cosine, **soft_cosine_params)
            else:
                sims = tfidf.get_similarities(
                    coll1_feats, coll2_feats, threshold=threshold)

        # - alignment-based
        elif method.startswith('alignment'):
            # get scorer
            if 'scorer' in method_params:
                scorer = method_params['scorer']
            else:
                scorer_class = method_params.get('scorer_class', 'ConstantScorer')
                if isinstance(scorer_class, str):
                    scorer_class = getattr(methods, scorer_class)
                scorer = scorer_class(
                    **dict(field=field, **method_params.get('scorer_params', {}))
                ).fit(coll1, coll2, embs=embs)
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
