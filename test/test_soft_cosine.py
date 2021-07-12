
import unittest

import numpy as np
import scipy.sparse

from retrieve.corpora import load_vulgate
from retrieve.data import Criterion, TextPreprocessor, FeatureSelector
from retrieve.embeddings import Embeddings
from retrieve.methods import Tfidf
from retrieve.methods.vsm.soft_cosine import soft_cosine_similarities
from retrieve.methods.vsm.soft_cosine import soft_cosine_simple
from retrieve.methods.vsm.soft_cosine import parallel_soft_cosine

from sklearn.metrics import pairwise_kernels


class TestSoftCosine(unittest.TestCase):
    def setUp(self):
        # load
        vulg = load_vulgate(max_verses=1000)
        # preprocess
        TextPreprocessor().process_collections(vulg, min_n=1, max_n=1)
        # drop features and get vocabulary
        fsel = FeatureSelector(vulg)
        vocab = fsel.filter_collections(
            vulg, criterion=(Criterion.DF >= 2) & (Criterion.FREQ >= 5))
        # get documents
        feats = vulg.get_features(cast=set)
        # transform to tfidf
        feats = Tfidf(vocab).fit(feats).transform(feats)
        query, index = feats[:feats.shape[0]//2], feats[feats.shape[0]//2:]
        # load embeddings, make sure S is in same order as vocab
        embs = Embeddings.from_resource(
            'latin.lemma.ft.dim100.mc2.embeddings.gz', vocab=vocab)

        self.embs = embs
        self.query = query
        self.index = index
        self.vocab = vocab

    def test_degenerate(self):
        # simple cosine similarity (we always return normalized vectors)
        sims1 = pairwise_kernels(self.query, self.index, metric='linear')
        # degenerate soft cosine should be equal to cosine
        sims2 = soft_cosine_similarities(
            self.query, self.index, np.identity(len(self.vocab)))

        self.assertTrue(np.allclose(sims1, sims2))

    def test_full(self):
        S = self.embs.get_S(vocab=self.vocab, fill_missing=True, cutoff=0.75, beta=2)
        sims = soft_cosine_similarities(self.query, self.index, S)
        for i in range(10):
            for j in range(10):
                sim1 = sims[i, j]
                sim2 = soft_cosine_simple(self.query[i], self.index[j], S)
                self.assertAlmostEqual(sim1, sim2, msg=f"{i}!={j}; {sim1}:{sim2}")

    def test_sparse(self):
        S = self.embs.get_S(vocab=self.vocab, fill_missing=True, cutoff=0.75, beta=2)
        self.assertTrue(
            scipy.sparse.issparse(S),
            msg="get_S returns sparse if fill_missing is True or cutoff > 0.0")
        sims2 = soft_cosine_similarities(self.query, self.index, S)
        self.assertTrue(
            scipy.sparse.issparse(sims2),
            msg="soft_cosine_similarities returns sparse if S is sparse")
        sims1 = soft_cosine_similarities(self.query, self.index, S.todense())
        self.assertFalse(
            scipy.sparse.issparse(sims1),
            msg="soft_cosine_similarities returns dense if S is dense")
        self.assertTrue(
            np.allclose(sims2.todense(), sims1),
            msg="sparse and dense results match")

    def test_parallel(self):
        S = self.embs.get_S(vocab=self.vocab, fill_missing=True, cutoff=0.75, beta=2)
        sims_parallel = parallel_soft_cosine(self.query, self.index, S, threshold=0.25)
        sims_single = soft_cosine_similarities(self.query, self.index, S, threshold=0.25)

        self.assertTrue(all(sims_parallel.data == sims_single.data))

