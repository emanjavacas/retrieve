
import uuid
import functools
import operator
import collections
from typing import Any, List, Dict, Tuple
import string
import re
import logging
from dataclasses import dataclass

import numpy as np

from retrieve import utils
from retrieve.methods import (jaccard, containment,
                              weighted_containment, weighted_jaccard)
from retrieve.methods import local_alignment, get_horizontal_alignment

logger = logging.getLogger(__name__)


@dataclass(eq=True, frozen=True)
class Ref:
    """
    Dataclass to store reference information. It's hashable and allows
    to incorporate metadata in the `meta` field as a tuple
    """
    source: Tuple[Any]
    target: Tuple[Any]
    meta: tuple = ()


class Doc:
    """
    Convenience class representing a document.

    Keeps any morphological tags passed to the constructor in memory
    as well as doc ids and refs

    Arguments
    =========
    fields : dict containing input
        must include a "token" field with the tokenized text
        currently it also assumes a "lemma" field
        "pos" can be used for further preprocessing
    doc_id : any document identifier
    ref : Ref
    """
    def __init__(self,
                 fields: Dict[str, List[any]],
                 doc_id: Any):

        if isinstance(doc_id, int):
            raise ValueError("Can't use `doc_id` of type integer")
        if 'token' not in fields:
            raise ValueError("`fields` requires 'token' data")

        self.fields = fields
        self.doc_id = doc_id
        self._features = []

    @property
    def text(self):
        return self.get_features(field='token')

    def to_counter(self, field=None):
        return collections.Counter(self.get_features(field=field))

    def get_features(self, field=None):
        if not field:
            if not self.features:
                raise ValueError("Unprocessed doc: [{}]".format(str(self.doc_id)))
            text = self.features
        else:
            text = self.fields[field]

        return text

    def __repr__(self):
        return '<Doc doc_id={} text="{}"/>'.format(
            str(self.doc_id), ' '.join(self.text)[:30] + "...")


def _wrap_fn(fn, use_counter=False):
    def wrapped(this, that, field='lemma', **kwargs):
        if use_counter:
            return fn(this.to_counter(field), that.to_counter(field), **kwargs)
        else:
            return fn(this.get_features(field), that.get_features(field), **kwargs)
    return wrapped


setattr(Doc, 'jaccard', _wrap_fn(jaccard, use_counter=True))
setattr(Doc, 'weighted_jaccard', _wrap_fn(weighted_jaccard, use_counter=True))
setattr(Doc, 'containment', _wrap_fn(containment, use_counter=True))
setattr(Doc, 'weighted_containment', _wrap_fn(weighted_containment, use_counter=True))
setattr(Doc, 'local_alignment', _wrap_fn(local_alignment))
setattr(Doc, 'get_horizontal_alignment', _wrap_fn(get_horizontal_alignment))


class Collection:
    """
    Class representing a collection of docs

    Arguments
    =========
    docs : list of Doc
    """
    def __init__(self, docs, name=None):
        self._docs = docs
        self._doc_ids = {doc.doc_id: idx for idx, doc in enumerate(docs)}
        # identifier for collection
        self.name = name or str(uuid.uuid4())[:8]

    def __getitem__(self, idx):
        if isinstance(idx, int):
            # access by index
            return self._docs[idx]
        # access by key
        return self._docs[self._doc_ids[idx]]

    def __len__(self):
        return len(self._docs)

    def __contains__(self, doc_id):
        return doc_id in self._doc_ids

    def get_doc_idx(self, doc_id):
        # access by doc id
        return self._doc_ids[doc_id]

    def get_docs(self, index=None):
        """
        Generator over docs

        Arguments
        =========
        index : list or set or dict of document indices to get (optional)
        """
        # TODO: index should probably be based on doc ids
        if index is not None:
            index = set(index)
        for idx, doc in enumerate(self._docs):
            if index is not None and idx not in index:
                continue
            yield doc

    def get_nonempty_features(self, **kwargs):
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
        for idx, feats in enumerate(self.get_features(**kwargs)):
            # handle empty documents
            if feats:
                output.append(feats)
                index.append(idx)

        return output, index

    def get_features(self, cast=None, min_features=0):
        """
        Get preprocessed text.

        Arguments
        =========
        cast : func (optonal), document features are casted with `cast` if passed

        Output
        ======
        list of lists with features
        """
        output = []
        for doc in self.get_docs():
            # get features
            feats = doc.features
            if cast is not None:
                feats = cast(feats)
            # empty input if number of features falls below threshold
            if min_features > 0 and len(feats) < min_features:
                feats = cast([])
            output.append(feats)
        return output

    def get_field_vocab(self, field):
        """
        Get vocabulary of a given input field (ignores preprocessing)
        """
        return collections.Counter(
            w for doc in self.get_docs() for w in doc.fields[field])


class TextPreprocessor:
    """
    Preprocess docs based on doc metadata
    """

    PUNCT = r"[{}]+".format(string.punctuation)

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

    def process(self, doc, verbose=False, **kwargs):
        """
        Process input text creating n-grams on the processed output.

        kwargs : get_ngrams additional arguments
        """
        filtered = []

        for i in range(len(doc.fields['token'])):
            target = doc.fields[self.field][i]

            if self.drop_punctuation and re.fullmatch(
                    TextPreprocessor.PUNCT, doc.fields[self.punct_field][i]):
                logger.debug("Dropping punctuation: {}".format(
                    doc.fields[self.punct_field][i]))
                continue

            if self.stopwords:
                if doc.fields[self.stop_field][i].lower() in self.stopwords:
                    logger.debug("Dropping stopword: {}".format(
                        doc.fields[self.stop_field][i]))
                    continue

            reg_match = True
            for re_field, regex in self.field_regexes.items():
                if not re.match(regex, doc.fields[re_field][i]):
                    reg_match = False
                    break
            if not reg_match:
                logger.debug("Dropping regex {}: {}".format(
                    re_field, doc.fields[re_field][i]))
                continue

            if (self.replace_unk or self.drop_unk):
                if 'lemma' in doc.fields and doc.fields['lemma'][i] == self.unk_token:
                    if self.replace_unk:
                        target = doc.fields[self.unk_field][i]
                    elif self.drop_unk:
                        logger.debug("Dropping unknown")
                        continue

            if self.lower:
                target = target.lower()

            filtered.append(target)

        return list(utils.get_ngrams(filtered, **kwargs))

    def process_collections(self, *colls, **kwargs):
        for coll in colls:
            for doc in coll.get_docs():
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
    def __init__(self, *colls):
        self.fitted = False
        self.features = {}
        self.freqs = []
        self.dfs = []
        self.ndocs = 0
        self.register(*colls)

    def register_text(self, text):
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

    def register(self, *colls):
        for coll in colls:
            for feats in coll.get_features():
                self.register_text(feats)
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

    def get_vocab(self, criterion=None):
        if not self.fitted:
            raise ValueError("Selector isn't fitted")

        id2ft = {idx: ft for ft, idx in self.features.items()}

        if criterion is None:
            return {id2ft[idx]: self.freqs[idx] for idx in id2ft}

        index = criterion.apply(self)
        index = sorted(index, key=self.freqs.__getitem__, reverse=True)

        return {id2ft[idx]: self.freqs[idx] for idx in index}

    def filter_collections(self, *colls, criterion=None):
        vocab = self.get_vocab(criterion)
        for coll in colls:
            for doc in coll.get_docs():
                doc.features = [ft for ft in doc.features if ft in vocab]
        return vocab

    def filter_texts(self, texts, criterion):
        vocab = self.get_vocab(criterion)
        for text in texts:
            yield [ft for ft in text if ft in vocab]
