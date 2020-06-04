

from retrieve.data import Doc, Collection


def load_vulgate(path='texts/vulgate.csv'):
    with open(path) as f:
        docs = []
        for idx, l in enumerate(f):
            if not l.strip():
                continue
            book, chapter, verse, tokens, pos, tt, pie = l.strip().split('\t')
            docs.append(
                Doc(fields={'token': tokens.split(), 'lemma': pie.split(), 'tt': tt},
                    doc_id=(book, chapter, verse)))

    return Collection(docs)
