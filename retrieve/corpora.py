
import os
import glob
import json
import collections
import warnings
import logging

from retrieve.data import Doc, Ref, Collection


VULGATE = "https://raw.githubusercontent.com/emanjavacas/cc-patrology/master/output/vulgate.csv"

logger = logging.getLogger(__name__)


def decode_ref(ref):
    book, chapter, verse = ref.split('_')
    book = ' '.join(book.split('-'))
    return book, chapter, verse


def encode_ref(ref):
    book, chapter, verse = ref
    return '-'.join(book.split()) + '_' + '_'.join([chapter, verse])


def read_testament_books(testament='new'):
    books = []
    with open('data/texts/{}-testament.books'.format(testament)) as f:
        for line in f:
            line = line.strip()
            books.append(line)
    return books


def read_bible(path, fields=('token', 'pos', '_', 'lemma'),
               max_verses=-1, sort_docs=True, verse_ids=None):
    with open(path) as f:
        docs = []
        for line in f:
            if not line.strip():
                continue
            if max_verses > 0 and len(docs) >= max_verses:
                break
            book, chapter, verse, *data = line.strip().split('\t')
            if verse_ids is not None and (book, chapter, verse) not in verse_ids:
                continue
            data = [field.split() for field in data]
            if len(data) != len(fields):
                raise ValueError(
                    "Expected {} metadata fields, but got {}. File: {}"
                    .format(len(fields), len(data), path))
            doc_id = book, chapter, verse
            try:
                docs.append(Doc(fields=dict(zip(fields, data)), doc_id=doc_id))
            except ValueError as e:
                warnings.warn("Ignoring document {}".format(' '.join(doc_id)))

    if not sort_docs:
        return docs

    books = {book: idx for idx, book in enumerate(
        read_testament_books('old') + read_testament_books('new'))}

    def key(doc):
        book, chapter, verse = doc.doc_id
        return books.get(book, len(books)), int(chapter), int(verse)

    return sorted(docs, key=key)


def load_vulgate(path='data/texts/vulgate.csv',
                 include_blb=False, split_testaments=False, max_targets=None,
                 # read_bible kwargs
                 **kwargs):

    if path == 'data/texts/vulgate.csv' and not os.path.isfile(path):
        # download from cc-patrology repository
        try:
            import urllib
            logger.info("Downloading vulgate...")
            urllib.request.urlretrieve(VULGATE, path)
        except:
            raise ValueError("Couldn't download vulgate")

    docs = read_bible(path, **kwargs)
    coll = Collection(docs, name=os.path.basename(path))
    if not split_testaments:
        return coll

    # split
    old, new = set(read_testament_books('old')), set(read_testament_books('new'))
    old_books, new_books = [], []
    for doc in docs:
        if doc.doc_id[0] in old:
            old_books.append(doc)
        elif doc.doc_id[0] in new:
            new_books.append(doc)
        else:
            warnings.warn("Missing book: {}".format(doc.doc_id[0]))

    # add refs
    old = Collection(old_books, name='Old Testament')
    new = Collection(new_books, name='New Testament')
    if not include_blb:
        return old, new

    refs = []
    for ref in load_blb_refs(max_targets=max_targets):
        source, target = [], []
        for r in ref['source']:
            if r not in new:
                warnings.warn("Couldn't find verse: new " + str(r))
                continue
            source.append(new.get_doc_idx(r))
        for r in ref['target']:
            if r not in old:
                warnings.warn("Couldn't find verse: old " + str(r))
                continue
            target.append(old.get_doc_idx(r))

        if len(source) == 0 or len(target) == 0:
            warnings.warn("Missing refs for expected pair" + str(ref))
            continue

        refs.append(Ref(tuple(source), tuple(target),
                        meta={'ref_type': ref['ref_type'],
                              'source': ref['source'],
                              'target': ref['target']}))

    return old, new, refs


def load_blb_refs(path='data/texts/blb.refs.json', max_targets=None):
    with open(path) as f:
        output = []
        for ref in json.load(f):
            source = [decode_ref(s_ref) for s_ref in ref['source']]
            target = [decode_ref(t_ref) for t_ref in ref['target']]
            ref_type = ref['ref_type']
            if max_targets is not None:
                if (len(source) > max_targets or len(target) > max_targets):
                    continue
            output.append({'source': source, 'target': target, 'ref_type': ref_type})

    return output


def read_doc(path, fields=('token', 'pos', '_', 'lemma')):
    output = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            try:
                data = line.strip().split('\t')
                if len(data) != len(fields):
                    raise ValueError(
                        "Expected {} metadata fields, but got {}. File: {}"
                        .format(len(fields), len(data), path))
                for key, val in zip(fields, data):
                    if key != '_':
                        output[key].append(val)
            except ValueError as e:
                raise e
            except Exception:
                print(line)

    return dict(output)


def read_refs(path):
    with open(path) as f:
        output = []
        for ref in json.load(f):
            ref['target'] = list(map(tuple, ref['target']))
            output.append(ref)
        return output


def shingle_doc(doc, f_id, overlap=10, window=20):
    shingled_docs = []
    n_words = len(doc[next(iter(doc.keys()))])
    for start in range(0, n_words, window - overlap):
        # doc might be smaller than window
        stop = min(start + window, n_words)
        assert start < stop, (start, stop, f_id)
        # doc id
        doc_id = f_id, (start, stop)
        # prepare doc
        fields = {key: vals[start:stop] for key, vals in doc.items()}
        fields['ids'] = list(range(start, stop))

        shingled_docs.append(Doc(fields=fields, doc_id=doc_id))

    return shingled_docs


def load_bernard(directory='data/texts/bernard',
                 bible_path='data/texts/vulgate.csv',
                 max_targets=None,
                 read_bible_kwargs={},
                 **kwargs):

    bible = Collection(read_bible(bible_path, **read_bible_kwargs), name='Vulgate')
    shingled_docs, shingled_refs = [], []
    for path in glob.glob(os.path.join(directory, '*.txt')):
        refs = read_refs(path.replace('.txt', '.refs.json'))
        doc = read_doc(path, fields=('w_id', 'token', 'pos', '_', 'lemma'))
        shingles = shingle_doc(doc, path, **kwargs)
        for ref in refs:
            source = []
            target = ref['target']
            if any(t not in bible for t in target):
                warnings.warn("Missing target")
                continue
            for shingle in shingles:
                if set(ref['source']).intersection(set(shingle.fields['ids'])):
                    source.append(shingle.doc_id)
            if source and (max_targets is None or len(source) <= max_targets) and \
               target and (max_targets is None or len(target) <= max_targets):
                shingled_refs.append(Ref(tuple(source), tuple(target), meta=ref))
            else:
                warnings.warn("Ignoring target")

        shingled_docs.extend(shingles)

    shingled_docs = Collection(shingled_docs, name='Bernard')

    return shingled_docs, bible, shingled_refs
