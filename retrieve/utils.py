
import time
import contextlib


def get_ngrams(s, min_n=1, max_n=1, sep='--'):
    """
    N-gram generator over input sequence. Allows multiple n-gram orders at once.
    """
    for n in range(min_n, max_n + 1):
        for ngram in zip(*[s[i:] for i in range(n)]):
            yield sep.join(ngram)


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


def load_stopwords(path):
    """
    Load stopwords from vertical format file. Ignore comments (#) and (?) doubted
    """
    stopwords = []
    with open(path) as f:
        for line in f:
            if line.startswith('?') or line.startswith('#'):
                continue
            stopwords.append(line.strip())
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
def timer(print_on_leave=True, desc='', fmt='took {:.2} secs in total', **kwargs):

    start = last = time.time()

    def time_so_far(desc='', fmt='took {:.2} secs', **kwargs):
        nonlocal last
        desc = ' - ' + (desc + ' ' if desc else desc)
        print(desc + fmt.format(time.time() - last), **kwargs)
        last = time.time()

    yield time_so_far

    desc = desc + ' ' if desc else desc

    print(desc + fmt.format(time.time() - start), **kwargs)
