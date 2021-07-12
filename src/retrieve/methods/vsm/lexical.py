
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD

from .base import VSM, VSMSoftCosine, init_sklearn_vectorizer


class Tfidf(VSMSoftCosine):
    def __init__(self, vocab, **kwargs):
        super().__init__(vocab, TfidfVectorizer, **kwargs)

    def fit(self, sents):
        self.vectorizer.fit(' '.join(s) for s in sents)
        return self

    def transform(self, sents):
        return self.vectorizer.transform(' '.join(s) for s in sents)


class BOW(VSMSoftCosine):
    def __init__(self, vocab, **kwargs):
        super().__init__(vocab, CountVectorizer, **kwargs)

    def fit(self, sents):
        self.vectorizer.fit(' '.join(s) for s in sents)
        return self

    def transform(self, sents):
        return self.vectorizer.transform(' '.join(s) for s in sents)


class LSI(VSM):
    def __init__(self, npc, vocab, vectorizer=CountVectorizer, **kwargs):
        self.npc = npc
        self.vectorizer = init_sklearn_vectorizer(
            vectorizer, vocabulary=vocab, **kwargs)
        self.svd = TruncatedSVD(n_components=npc)

    def fit(self, sents):
        self.vectorizer.fit(' '.join(s) for s in sents)
        return self

    def transform(self, sents):
        return self.svd.fit_transform(
            self.vectorizer.transform(
                ' '.join(s) for s in sents))
