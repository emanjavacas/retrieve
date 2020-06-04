
import scipy.sparse

from retrieve.data import Criterion, TextPreprocessor, FeatureSelector
from retrieve.corpora import load_vulgate
from retrieve.set_similarity import SetSimilarity
from retrieve.compare.align import create_embedding_scorer
from retrieve import utils

vulg = load_vulgate()
stops = utils.load_stopwords('all.stop')
TextPreprocessor(stopwords=None).process_collection(vulg, min_n=2, max_n=4)
fsel = FeatureSelector().update_collection_stats(vulg)
fsel.filter_collection(vulg, Criterion.DF >= 2)
embs = utils.Embeddings.from_csv(
    'latin.lemma.embeddings', vocab=vulg.get_field_vocab('lemma'))
scorer = create_embedding_scorer(embs)
features, doc_ids = vulg.get_nonempty_features(cast=set)
print("Lost", len(vulg) - len(doc_ids))
S = SetSimilarity(0.5, similarity_fn="jaccard").get_similarities(features)
x, y, sims = scipy.sparse.find(S)
x, y, sims = x[:100], y[:100], sims[:100]
index = (sims < 0.9)

for idx, (i, j) in enumerate(zip(x[index], y[index])):
    s1, s2 = vulg[doc_ids[i]], vulg[doc_ids[j]]
    print(s1)
    print(s2)
    print(" * ", sims[index][idx])
    print(" * ", s1.containment(s2, field=None))
    a1, a2, score = s1.local_alignment(s2, scorer=scorer)
    print(" * ", score)
    print('\n'.join(s1.get_horizontal_alignment(s2)))
    print()
    print("---")

from retrieve.compare import align_collections

output = align_collections(vulg, vulg, 'lemma', S=S)
