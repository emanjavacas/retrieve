README
======

# `data/stop/latin.stop`

Filtered from the top 1k lemmas found in the cc-patrology corpus.

# `data/stop/greek_berra.stop`

Gratefully taken from Aurelien Berra repository https://github.com/aurelberra/stopwords

wget https://raw.githubusercontent.com/aurelberra/stopwords/master/stopwords_greek.json
cat stopwords_greek.json | jq -r '{ a:.ADVERBS, b:.ARTICLES, c:.CONJUNCTIONS, d:.INTERJECTIONS, e:.["PREPOSITIONS/POSTPOSITIONS"] } | .[] | .[]' > data/stop/greek_berra.stop
cat stopwords_greek.json | jq -r '.PRONOUNS | keys | .[]' >> data/stop/greek_berra.stop
cat stopwords_greek.json | jq -r '.VERBS | keys | .[]' >> data/stop/greek_berra.stop

# `data/stop/latin_berra.stop`

wget https://raw.githubusercontent.com/aurelberra/stopwords/master/stopwords_latin.json
cat stopwords_latin.json | jq -r '.["ADJECTIVES"] | .[] | .[]' > data/stop/latin_berra.stop
cat stopwords_latin.json | jq -r '.["PRONOUNS"] | .[] | .[]' >> data/stop/latin_berra.stop
cat stopwords_latin.json | jq -r '.["VERBS"] | keys | .[]' >> data/stop/latin_berra.stop
cat stopwords_latin.json | jq -r '.["ADVERBS"] | .[] ' >> data/stop/latin_berra.stop
cat stopwords_latin.json | jq -r '.["CONJUNCTIONS"] | .[] ' >> data/stop/latin_berra.stop
cat stopwords_latin.json | jq -r '.["PREPOSITIONS"] | .[] ' >> data/stop/latin_berra.stop

# `data/stop/english.stop`

Gratefully taken from NLTK

wget https://raw.githubusercontent.com/igorbrigadir/stopwords/master/en/nltk.txt
mv nltk.txt data/stop/english.stop
