
import collections
import dataclasses as dc
from typing import Any, List
import string
import re

import numpy as np

# regexes
PUNCT = r"[{}]+".format(string.punctuation)


def containment(s1, s2):
    s1, s2 = set(s1), set(s2)
    return len(s1 & s2) / (len(s1) or 1)


def jaccard(s1, s2):
    s1, s2 = set(s1), set(s2)
    return len(s1.intersection(s2)) / len(s1.union(s2))


def weighted_jaccard(s1, s2):
    den = num = 0
    for w in set(s1).union(s2):
        c1, c2 = s1.get(w, 0), s2.get(w, 0)
        num += min(c1, c2)
        den += max(c1, c2)
    return num / den


def weighted_containment(s1, s2):
    den = num = 0
    for w in set(s1).union(s2):
        c1, c2 = s1.get(w, 0), s2.get(w, 0)
        num += min(c1, c2)
        den += c2
    return num / den


def _wrap_fn(fn):
    def wrapped(this, other, field='token'):
        return fn(this.to_counter(field), other.to_counter(field))
    return wrapped


def to_counter(doc, field=None):
    if not field:
        if not doc.text:
            raise ValueError(
                "Doc [{}] needs to be processed if field is not passed".format(
                    doc.doc_id))
        text = doc.text
    else:
        text = doc.fields[field]
    return collections.Counter(text)


_fns = [jaccard,
        weighted_jaccard,
        containment,
        weighted_containment]


Doc = dc.make_dataclass(
    'Doc',
    [('fields', dict),
     ('doc_id', Any),
     ('ref', Any, dc.field(default=None)),
     ('text', List[str], dc.field(default=None))],
    namespace={'to_counter': to_counter,
               '',
               **{fn.__name__:  _wrap_fn(fn) for fn in _fns}})


class TextPreprocessor:
    def __init__(self,
                 target_field='lemma',
                 lower=True,
                 field_regexes={},
                 drop_punctuation=True, punct_field='token',
                 replace_unk=False, unk_token='$unk', unk_field='token',
                 stopwords=set(), stopword_field='lemma'):

        self.target_field = target_field
        self.lower = lower
        # punctuation
        self.drop_punctuation = drop_punctuation
        self.punct_field = punct_field
        # unks
        self.replace_unk = replace_unk
        self.unk_token = unk_token
        self.unk_field = unk_field
        # stopwords
        self.stopwords = stopwords
        self.stopword_field = stopword_field
        # regexes
        self.field_regexes = field_regexes

    def process(self, doc, verbose=False):
        filtered = []
        f2i = {field: idx for idx, field in enumerate(sorted(doc.fields))}

        for fs in zip(*[doc.fields[field] for field in sorted(f2i)]):
            target = fs[f2i[self.target_field]]

            # ignore punctuation
            if self.drop_punctuation and re.fullmatch(PUNCT, fs[f2i[self.punct_field]]):
                if verbose:
                    print("punctuation: {}".format(fs[f2i[self.punct_field]]))
                continue

            # ignore stopwords
            if self.stopwords:
                if fs[f2i[self.stopword_field]].lower() in self.stopwords:
                    if verbose:
                        print("stopword", fs[f2i[self.stopword_field]])
                    continue

            # unks
            if fs[f2i['lemma']] == self.unk_token:
                if self.replace_unk:
                    target = fs[f2i[self.unk_field]]
                elif self.drop_unk:
                    if verbose:
                        print("unknown", target)
                    continue

            # regexes
            for re_field in self.field_regexes:
                if not re.match(self.field_regexes[re_field], fs[f2i[re_field]]):
                    if verbose:
                        print("regex {}".format(re_field), fs[f2i[re_field]])
                    continue

            if self.lower:
                target = target.lower()

            filtered.append(target)

        return filtered


class FeatureSelector:
    def __init__(self, ):
        self

    def collect(self, text):
        pass

    def filter(self, ):
        pass


class TfIdf:
    def __init__(self, min_df=2, prune_min=0.0):
        """
        min_df : int, minimum document frequency
        prune_min : float, 
            prune features based on tf-idf lower than `prune_min` quantile

        """
        self.min_df = min_df
        self.prune_min = prune_min
        self.tf = collections.Counter()
        self.df = collections.Counter()
        self.n_docs = 0
        self.feats = None

    def fit_transform(self, texts):
        for text in texts:
            counts = collections.Counter(text)
            self.n_docs += 1
            self.df.update(list(counts))
            self.tf.update(counts)

        feats = list(self.tf)
        tf = np.array(list(map(self.tf.get, feats)))
        df = np.array(list(map(self.df.get, feats)))

        if self.min_df > 1:
            index, = np.where(df >= self.min_df)
            tf, df = tf[index], df[index]
            feats = [feats[i] for i in index]        

        tf = np.log(1 + tf)
        idf = np.log(self.n_docs / df)

        if self.prune_min > 0:
            tfidf = tf * idf
            index, = np.where(tfidf > np.quantile(tfidf, self.prune_min))
            feats = [feats[i] for i in index]

        self.feats = {ft: idx for idx, ft in enumerate(feats)}

        return self

    def transform(self, text):
        if self.n_docs == 0:
            raise ValueError("TfIdf hasn't been fit")

        return [ft for ft in text if ft in self.feats]


def get_ngrams(s, min_n=1, max_n=1, sep='#$#'):
    for n in range(min_n, max_n + 1):
        for ngram in zip(*[s[i:] for i in range(n)]):
            yield sep.join(ngram)


def create_corpus(docs, processor, tfidf=None, min_n=1, max_n=1):
    
    for doc in docs:
        doc.text = list(get_ngrams(processor.process(doc), min_n=min_n, max_n=max_n))
        # update stats

    # feature selection
    pass


with open('../latin-data/output/processed/vatican.dev') as f:
    docs = []
    for idx, l in enumerate(f):
        if not l.strip():
            continue
        tokens = l.strip().split()
        docs.append(Doc(fields={'token': tokens, 'lemma': tokens}, doc_id=idx))

create_corpus(docs, TextPreprocessor(), max_n=1)

from sklearn.feature_extraction.text import TfidfVectorizer

# def select_tfidf(docs, min_df=2, prune_min=0.0, **kwargs):
tfidf = TfidfVectorizer(lowercase=False, min_df=2, token_pattern=r'[^ ]+')
X = tfidf.fit_transform([' '.join(doc.processed) for doc in docs])
scores = np.squeeze(np.array(X.sum(0)))

import collections
counter = collections.Counter(w for doc in docs for w in doc.processed)

# sorted(list(zip(scores, [counter[w] for w in tfidf.get_feature_names()], tfidf.get_feature_names())))[-100:]

by_tfidf = sorted(list(zip(scores, tfidf.get_feature_names())), reverse=True)
by_freqs = [(cnt, w) for w, cnt in counter.most_common()]

cutoff = 200
set([w for _, w in by_tfidf[:cutoff]]).difference(
    set([w for _, w in by_freqs[:cutoff]]))

