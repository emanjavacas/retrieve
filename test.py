
# import numpy as np
# import scipy.sparse

# from retrieve.data import Criterion, TextPreprocessor, FeatureSelector
# from retrieve.corpora import load_vulgate
# from retrieve.set_similarity import SetSimilarity
# from retrieve.compare.align import create_embedding_scorer
# from retrieve.compare import align_collections
# from retrieve import utils

# # # load data
# vulg = load_vulgate(include_blb=True)
# stops = utils.load_stopwords('all.stop')
# TextPreprocessor(stopwords=stops).process_collection(vulg, min_n=1, max_n=1)
# fsel = FeatureSelector(vulg)
# vocab = fsel.filter_collection(vulg, (Criterion.DF >= 2) & (Criterion.FREQ >= 5))
# embs = utils.Embeddings.from_csv('latin.lemma.embeddings', vocab=vocab)

# scorer = create_embedding_scorer(embs)
# features, doc_ids = vulg.get_nonempty_features(cast=set)
# # print("Lost", len(vulg) - len(doc_ids))
# # fn = 'jaccard'
# # S = SetSimilarity(0.5, similarity_fn=fn).get_similarities(features)
# # x, y, sims = scipy.sparse.find(S)
# # index = (sims < 0.9)

# # for idx, (i, j) in enumerate(zip(x[index], y[index])):
# #     s1, s2 = vulg[doc_ids[i]], vulg[doc_ids[j]]
# #     print(s1)
# #     print(s2)
# #     print(" * ", sims[index][idx])
# #     print(" * ", s1.containment(s2, field=None))
# #     a1, a2, score = s1.local_alignment(s2, scorer=scorer)
# #     print(" * ", score)
# #     print('\n'.join(s1.get_horizontal_alignment(s2)))
# #     print()
# #     print("---")

# # output = align_collections(vulg, vulg, 'lemma', S=S, processes=1)

# from retrieve.vsm.lexical import Tfidf

# feats, _ = list(vulg.get_nonempty_features())
# queries, index = feats[:1000], feats[1000:2000]
# vsm = Tfidf(vocab).fit(feats)
# # list(vocab) == vsm.vectorizer.get_feature_names()
# vsm.get_soft_cosine_S(queries, index, embs, cutoff=0.8)
# S = vsm.get_similarities(feats, feats, metric='linear')
# S2 = vsm.get_soft_cosine_S(feats, feats, embs)
# queries, index = feats, feats
# beta = 1
# S_metric = 'cosine'
# index, queries = list(index), list(queries)
# transform = vsm.transform(index + queries)
# index, queries = transform[:len(index)], transform[:len(index)]
# keys, S = embs.get_S(words=vsm.vectorizer.get_feature_names(),
#                      metric=S_metric, fill_missing=True)
# words = vsm.vectorizer.get_feature_names()
# M = np.power(np.clip(S, a_min=0, a_max=np.max(S)), 5)
# I = np.zeros_like(M)
# np.fill_diagonal(I, 1)

# SC = soft_cosine_similarities(queries[:11], index[:11], S)
# from retrieve.compare import soft_cosine_similarities
# S_dense = embs.get_S(words=vocab, fill_missing=True)
# S_sparse = embs.get_S(words=vocab, fill_missing=True, cutoff=0.75)

# a=soft_cosine_similarities(queries, index, scipy.sparse.lil_matrix(S_sparse))
# b=soft_cosine_similarities(queries, index, S_sparse)

# import timeit

# a=timeit.Timer(lambda: soft_cosine_similarities(queries, index, scipy.sparse.lil_matrix(S_sparse))).timeit(number=5)
# b=timeit.Timer(lambda: soft_cosine_similarities(queries, index, S_sparse)).timeit(number=5)

