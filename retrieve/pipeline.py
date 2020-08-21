
from retrieve import utils
from retrieve.data import Criterion, TextPreprocessor, FeatureSelector
from retrieve.embeddings import Embeddings
from retrieve.methods import SetSimilarity, Tfidf, align_collections
from retrieve.methods import create_embedding_scorer, ConstantScorer


def require_embeddings(embs, msg='', **kwargs):
    if isinstance(embs, str):
        embs = Embeddings.from_file(embs, **kwargs)
    if not isinstance(embs, Embeddings):
        raise ValueError(msg)
    return embs


def pipeline(coll1, coll2=None,
             # Text Preprocessing
             field='lemma', lower=True, stopwords=None, stop_field='lemma',
             # Ngrams
             min_n=1, max_n=1, skip_k=0, sep='--',
             # Feature Selection
             criterion=None,
             method='set-based', threshold=0, processes=-1, embs=None,
             # Set-based
             # - SetSimilarity: similarity_fn
             #     ('containment', 'containment_min', 'jaccard')
             # VSM-based
             # - Tfidf: vocab, **sklearn,feature_extraction.text.TfidfVectorizer
             # Alignment-based
             # - match, mismatch, open_gap, extend_gap, cutoff, factor
             method_fit={},
             # Soft_cosine_params: cutoff, beta
             use_soft_cosine=False, soft_cosine_params={},
             # For Blast-style alignment
             precomputed_sims=None,
             # return time stats
             return_stats=False):

    colls = [coll for coll in [coll1, coll2] if coll]

    if isinstance(stopwords, str):
        stopwords = utils.Stopwords(stopwords)

    stats = {}

    with utils.timer() as timer:
        # preprocessing
        TextPreprocessor(
            field=field, lower=lower, stopwords=stopwords, stop_field=stop_field,
        ).process_collections(*colls, min_n=min_n, max_n=max_n, skip_k=skip_k, sep=sep)
        fsel = FeatureSelector(*colls)
        vocab = fsel.filter_collections(*colls, criterion=criterion)

        stats['preprocessing'] = timer(desc='Preprocessing')

        # similarities
        if method == 'set-based':
            coll1_feats = coll1.get_features(cast=set)
            coll2_feats = coll2.get_features(cast=set) if coll2 else coll1_feats
            sims = SetSimilarity(threshold, **method_fit).get_similarities(
                coll1_feats, coll2_feats, processes=processes)
        elif method == 'vsm-based':
            coll1_feats = coll1.get_features()
            coll2_feats = coll2.get_features() if coll2 is not None else coll1_feats
            tfidf = Tfidf(vocab, **method_fit).fit(coll1_feats + coll2_feats)
            if use_soft_cosine:
                embs = require_embeddings(
                    embs, msg='soft cosine requires embeddings', vocab=vocab)
                sims = tfidf.get_soft_cosine_similarities(
                    coll1_feats, coll2_feats, embs=embs,
                    threshold=threshold, **soft_cosine_params)
            else:
                sims = tfidf.get_similarities(
                    coll1_feats, coll2_feats, threshold=threshold)
        elif method == 'alignment-based':
            if embs is not None:
                embs = require_embeddings(embs, vocab=fsel.get_vocab())
                scorer = create_embedding_scorer(
                    embs,
                    **{key: val for key, val in method_fit.items()
                       if key in set(['match', 'mismatch', 'cutoff', 'factor'])})
            else:
                scorer = ConstantScorer(
                    **{key: val for key, val in method_fit.items()
                       if key in set(['match', 'mismatch'])})
            if precomputed_sims is not None:
                print("Computing {} alignments...".format(precomputed_sims.nnz))
            sims = align_collections(
                coll1, coll2,
                S=precomputed_sims, field=None, processes=processes, scorer=scorer,
                **{key: val for key, val in method_fit.items()
                   if key in set(['extend_gap', 'open_gap'])})
        else:
            raise ValueError("Unknown method", method)

        stats['similarity'] = timer(desc='Similarity')

    if return_stats:
        return sims, stats

    return sims
