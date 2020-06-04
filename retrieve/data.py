
import functools
import operator
import collections
from typing import Any, List, Dict
import string
import re
import logging

import numpy as np

from retrieve.set_similarity import (jaccard, containment,
                                     weighted_containment, weighted_jaccard)
from retrieve import utils
from retrieve.compare.align import local_alignment, get_horizontal_alignment


logging.basicConfig(
    format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)


# regexes
PUNCT = r"[{}]+".format(string.punctuation)


class Doc:
    def __init__(self,
                 fields: Dict[str, List[any]],
                 doc_id: Any,
                 ref: Any = None):

        self.fields = fields
        self.doc_id = doc_id
        self.ref = ref
        self._features = []

    def to_counter(self, field=None):
        if not field:
            if not self.features:
                raise ValueError(
                    "Doc [{}] needs to be processed if field is not passed".format(
                        self.doc_id))
            text = self.features
        else:
            text = self.fields[field]
        return collections.Counter(text)

    def to_text(self, field='token'):
        return ' '.join(self.fields[field])

    def __repr__(self):
        return '<Doc doc_id={} ref={} text="{}"/>'.format(
            str(self.doc_id), str(self.ref), self.to_text()[:30] + "...")


def _wrap_fn(fn, use_counter=False):
    def wrapped(this, other, field='lemma', **kwargs):
        if use_counter:
            return fn(this.to_counter(field), other.to_counter(field), **kwargs)
        else:
            return fn(this.fields[field], other.fields[field], **kwargs)
    return wrapped


setattr(Doc, 'jaccard', _wrap_fn(jaccard, use_counter=True))
setattr(Doc, 'weighted_jaccard', _wrap_fn(weighted_jaccard, use_counter=True))
setattr(Doc, 'containment', _wrap_fn(containment, use_counter=True))
setattr(Doc, 'weighted_containment', _wrap_fn(weighted_containment, use_counter=True))
setattr(Doc, 'local_alignment', _wrap_fn(local_alignment))
setattr(Doc, 'get_horizontal_alignment', _wrap_fn(get_horizontal_alignment))


class Collection:
    def __init__(self, docs):
        self._docs = docs

    def __getitem__(self, idx):
        return self._docs[idx]

    def __len__(self):
        return len(self._docs)

    def get_docs(self):
        """
        Generator over docs
        """
        yield from self._docs

    def get_nonempty_features(self, cast=None):
        """
        Avoid getting empty bags of features. Documents might
        be empty after preprocessing or feature selection.

        Output
        ======
        Tuple of (features, index):
            features, list of document features
            index, list of indices mapping the document idx in the original collection
        """
        output, index = [], []
        for idx, feats in enumerate(self.get_features(cast=cast)):
            # handle empty documents
            if feats:
                output.append(feats)
                index.append(idx)

        return output, index

    def get_features(self, cast=None):
        """
        Get preprocessed text.

        Arguments
        =========
        cast : func (optonal), document features are casted with `cast` if passed

        Output
        ======
        A generator, if only_nonempty is False (default), otherwise a tup
        """
        for doc in self.get_docs():
            # get features
            feats = doc.features
            if cast is not None:
                feats = cast(feats)
            yield feats

    def get_field_vocab(self, field):
        """
        Get vocabulary of a given input field (ignores preprocessing)
        """
        return collections.Counter(
            w for doc in self.get_docs() for w in doc.fields[field])


class TextPreprocessor:
    def __init__(self,
                 field='lemma',
                 lower=True,
                 field_regexes={},
                 drop_punctuation=True, punct_field='token',
                 replace_unk=False, drop_unk=False, unk_token='$unk$', unk_field='token',
                 stopwords=set(), stop_field='lemma'):

        self.field = field
        self.lower = lower
        # punctuation
        self.drop_punctuation = drop_punctuation
        self.punct_field = punct_field
        # unks
        self.replace_unk = replace_unk
        self.drop_unk = drop_unk
        self.unk_token = unk_token
        self.unk_field = unk_field
        # stopwords
        self.stopwords = stopwords
        self.stop_field = stop_field
        # regexes
        self.field_regexes = field_regexes

    def process(self, doc, verbose=False, min_n=1, max_n=1):
        filtered = []

        for i in range(len(doc.fields['token'])):
            target = doc.fields[self.field][i]

            if self.drop_punctuation and re.fullmatch(
                    PUNCT, doc.fields[self.punct_field][i]):
                logging.debug("Dropping punctuation: {}".format(
                    doc.fields[self.punct_field][i]))
                continue

            if self.stopwords:
                if doc.fields[self.stop_field][i].lower() in self.stopwords:
                    logging.debug("Dropping stopword: {}".format(
                        doc.fields[self.stop_field][i]))
                    continue

            reg_match = True
            for re_field, regex in self.field_regexes.items():
                if not re.match(regex, doc.fields[re_field][i]):
                    reg_match = False
                    break
            if not reg_match:
                logging.debug("Dropping regex {}: {}".format(
                    re_field, doc.fields[re_field][i]))
                continue

            if (self.replace_unk or self.drop_unk):
                if 'lemma' in doc.fields and doc.fields['lemma'][i] == self.unk_token:
                    if self.replace_unk:
                        target = doc.fields[self.unk_field][i]
                    elif self.drop_unk:
                        logging.debug("Dropping unknown")
                        continue

            if self.lower:
                target = target.lower()

            filtered.append(target)

        return list(utils.get_ngrams(filtered, min_n=min_n, max_n=max_n))

    def process_collection(self, collection, **kwargs):
        for doc in collection.get_docs():
            doc.features = self.process(doc, **kwargs)


class MetaCriterion(type):
    @property
    def DF(cls):
        return cls("DF")

    @property
    def FREQ(cls):
        return cls("FREQ")

    @property
    def IDF(cls):
        return cls("IDF")


class Criterion(object, metaclass=MetaCriterion):
    def __init__(self, field):
        self.field = field
        self.ops = []
        self.fields_, self.ops_ = [], []

    def _get_index(self, stats, val, operator):
        if val < 1 or (val == 1 and isinstance(val, float)):
            # convert stats to normalized ranks
            # assumes stats has already been argsorted
            stats = np.linspace(0, 1, len(stats))[stats.argsort().argsort()]

        index, = np.where(operator(stats, val))
        return index

    def apply(self, f_sel):
        if not self.ops:
            raise ValueError("Criterion not set until comparison")

        # concat current field to extra fields
        fields = [self.field] + self.fields_
        ops = [self.ops] + self.ops_
        stats = {f: f_sel._get_stats(f) for f in set(fields)}

        index = []
        for field, ops in zip(fields, ops):
            for val, op in ops:
                index.append(self._get_index(stats[field], val, op))

        if len(index) == 1:
            index = index[0]
        else:
            index = functools.reduce(np.intersect1d, index)

        return index

    def __and__(self, other):
        if not self.ops or not other.ops:
            raise ValueError("Criterion not set until comparison")

        self.fields_ += [other.field]
        self.ops_ += [other.ops]
        return self

    def __le__(self, val):
        self.ops.append((val, operator.le))
        return self

    def __lt__(self, val):
        self.ops.append((val, operator.lt))
        return self

    def __ge__(self, val):
        self.ops.append((val, operator.ge))
        return self

    def __gt__(self, val):
        self.ops.append((val, operator.gt))
        return self

    def __eq__(self, val):
        self.ops.append((val, operator.eq))
        return self


class FeatureSelector:
    def __init__(self):
        self.fitted = False
        self.features = {}
        self.freqs = []
        self.dfs = []
        self.ndocs = 0

    def update_stats(self, text):
        for ft, cnt in collections.Counter(text).items():
            idx = self.features.get(ft, len(self.features))
            if ft not in self.features:
                self.features[ft] = idx
                self.freqs.append(cnt)
                self.dfs.append(1)
            else:
                self.freqs[idx] += cnt
                self.dfs[idx] += 1
        self.ndocs += 1

    def update_collection_stats(self, *colls):
        for coll in colls:
            for feats in coll.get_features():
                self.update_stats(feats)
        self._fit()

        return self

    def _fit(self):
        assert len(self.features) == len(self.freqs) == len(self.dfs)
        self.freqs, self.dfs = np.array(self.freqs), np.array(self.dfs)
        self.fitted = True

    def _get_stats(self, field):
        if not self.fitted:
            raise ValueError("Selector isn't fitted")

        if field == "FREQ":
            return self.freqs
        elif field == "DF":
            return self.dfs
        elif field == "IDF":
            return np.log(np.sum(self.freqs) / self.dfs)
        else:
            raise ValueError("Requested unknown stats")

    def get_vocab(self, criterion):
        if not self.fitted:
            raise ValueError("Selector isn't fitted")

        id2ft = {idx: ft for ft, idx in self.features.items()}
        index = criterion.apply(self)
        index = sorted(index, key=self.freqs.__getitem__, reverse=True)

        return {id2ft[idx]: self.freqs[idx] for idx in index}

    def filter_collection(self, collection, criterion):
        vocab = self.get_vocab(criterion)
        for doc in collection.get_docs():
            doc.features = [ft for ft in doc.features if ft in vocab]

    def filter_texts(self, texts, criterion):
        vocab = self.get_vocab(criterion)
        for text in texts:
            yield [ft for ft in text if ft in vocab]


if __name__ == '__main__':
    import time
    from retrieve.corpora import load_vulgate

    collection = load_vulgate()
    stops = utils.load_stopwords('all.stop')
    processor = TextPreprocessor(stopwords=stops, field_regexes={'token': '[a-z]+'})
    start = time.time()
    processor.process_collection(collection)
    print(time.time() - start)
    fsel = FeatureSelector().update_collection_stats(collection)
    fsel.get_vocab((0.5 <= Criterion.FREQ <= 0.95))
    fsel.get_vocab(Criterion.FREQ >= 0.95)

    # test threshold
    for th_min, th_max in zip(range(1, 1000, 100), range(100, 10000, 1000)):
        vocab = fsel.get_vocab(th_min <= Criterion.DF < th_max)
        assert all([fsel.dfs[fsel.features[ft]] >= th_min for ft in vocab])
        assert all([fsel.dfs[fsel.features[ft]] < th_max for ft in vocab])
