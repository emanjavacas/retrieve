
import os
import logging
import time
import contextlib
import itertools
import unicodedata
import sys
if sys.version_info.minor < 7:
    from importlib_resources import open_text, is_resource
else:
    from importlib.resources import open_text, is_resource


logger = logging.getLogger(__name__)


class Ngrams:
    def __init__(self, min_n=1, max_n=1, skip_k=0, sep='--'):
        self.min_n = min_n
        self.max_n = max_n
        self.skip_k = skip_k
        self.sep = sep

    def get_summary(self):
        return (self.min_n, self.max_n, self.skip_k)

    def get_ngrams_generator(self, s):
        for ngram in get_ngrams(
                s,
                min_n=self.min_n,
                max_n=self.max_n,
                skip_k=self.skip_k):
            yield self.sep.join(ngram)

    def get_ngrams(self, s):
        return list(self.get_ngrams_generator(s))


def get_ngrams(s, min_n=1, max_n=1, skip_k=0):
    """
    N-gram generator over input sequence. Allows multiple n-gram orders at once
    as well as skip-grams
    """
    for n in range(min_n, max_n + 1):
        if skip_k and n > 1:
            for ngram in zip(*[s[i:] for i in range(n + skip_k)]):
                first, rest = ngram[:1], ngram[1:]
                for tail in itertools.combinations(rest, n - 1):
                    yield first + tail
        else:
            for ngram in zip(*[s[i:] for i in range(n)]):
                yield ngram


def chunks(it, size):
    """
    Chunk a generator into a given size (last chunk might be smaller)
    """
    buf = []
    for s in it:
        buf.append(s)
        if len(buf) == size:
            yield buf
            buf = []
    if len(buf) > 0:
        yield buf


def drop_string_diacritics(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


class Stopwords:
    def __init__(self, *paths, drop_diacritics=False):
        self.words = set()
        for path in paths:
            self.words.update(load_stopwords(path, drop_diacritics))
        self.paths = paths
        self.drop_diacritics = drop_diacritics

    def get_summary(self):
        return {'paths': self.paths, 'drop_diacritics': self.drop_diacritics}

    def __contains__(self, w):
        return w in self.words


def load_stopwords(path, drop_diacritics=False):
    """
    Load stopwords from vertical format file. Ignore comments (#) and (?) doubted
    """
    if os.path.isfile(path):
        with open(path) as f:
            lines = list(f)
    elif is_resource('retrieve.resources.stop', path):
        # try with package resource
        lines = list(open_text('retrieve.resources.stop', path))
    else:
        raise ValueError("Couldn't find file: '{}'".format(path))

    stopwords = []
    for line in lines:
        line = line.strip()
        if line.startswith('?') or line.startswith('#'):
            continue
        if not line:
            continue
        if drop_diacritics:
            line = drop_string_diacritics(line)
        stopwords.append(line)

    return set(stopwords)


def load_freqs(path, top_k=0):
    """
    Load frequencies from file in format:

        word1 123
        word2 5
    """
    freqs = {}
    with open(path) as f:
        for line in f:
            count, w = line.strip().split()
            freqs[w] = int(count)
            if top_k > 0 and len(freqs) >= top_k:
                break
    return freqs


@contextlib.contextmanager
def timer(print_on_leave=True, desc='', fmt='took {:0.1f} secs in total', **kwargs):

    start = last = time.time()

    def time_so_far(desc='', fmt='took {:0.1f} secs', **kwargs):
        nonlocal last
        desc = ' - ' + (desc + ' ' if desc else desc)
        took = time.time() - last
        logger.info(desc + fmt.format(took), **kwargs)
        last = time.time()
        return took

    yield time_so_far

    desc = desc + ' ' if desc else desc

    logger.info(desc + fmt.format(time.time() - start), **kwargs)
