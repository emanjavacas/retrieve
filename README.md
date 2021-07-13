RETRIEVE: A Text Reuse Software Package
--------
 
RETRIEVE is designed with the goal of making accessible a number of text reuse retrieval algorithmic paradigms (more concretely, three paradigms) to a broader audience. 


The focus of RETRIEVE is small and medium scale retrieval problems (collections in the order of **tens of thousands** to **hundreds of thousands** of sentences). However, the software tries to optimize memory space when possible, and with some manual tuning (and waiting time) it could eventually be ran on larger problems.


As of today, RETRIEVE implements the retrieval of text reuse around words. While it would be easy to extend it so that subword strings are used instead, this has been left out for now from the current implementation (PRs welcome).


Typically, text reuse retrieval software improves results by using lemmatized input. RETRIEVE tries to do a specialized thing well, and, therefore, does not include lemmatization in the pipeline. This means that you will have to provide lemmatized input, if you want to profit from the benefits of lemmatization. If you don't have a lemmatizer available for your language of choice, I recommend using [PIE](https://github.com/emanjavacas/pie/), which you can use to train your own lemmatizer.

# Installation

## Installing from PyPI

RETRIEVE can be installed from PyPI using `pip`. Just fire up a command prompt and type:

```pip install retrieve```

## Installation from source

RETRIEVE can also be installed by first downloading the repository, installing the dependencies and issuing the `python setup.py install` command within the top directory.

Dependencies are kept in the `requirements.txt` file. To install them, use:

```pip install -r requirements.txt```

# Workflow

The workflow consists of the following steps:
- Data Preparation: gathering sentences on which to carry out the text reuse search
- Text Preprocessing: processing input documents so as to facilitate the subsequent search
- Search: running search algorithms on the input collections

## Data preparation

RETRIEVE doesn't offer many tools in order to aid the data preparation process, but just functionality to load the resources and operate on them. The most important resources is a lemmatizer (see remarks in the Introduction), and, eventually, the curation of stopword lists. Additionally, the subsequence text preprocessing can be improved if POS-tags are available.

Loading is done with the `Collection` class, which is the appropriate input format for the search algorithms implemented in RETRIEVE.

A `Collection` is built around individual `Doc` instances. A `Doc` is just a data structure that holds the input text, as well as a document id and some textual metadata if available. A `Collection` can be loaded using the `Collection.from_file` and `Collection.from_csv` methods, or can be manually instantiated by manually creating indivual `Doc` instances and passing them to the `Collection` constructor.

```python
from retrieve.data import Doc, Collection
line1 = ['The', 'cat', 'sat', 'on', 'the', 'mat', '.']
line2 = ['The', 'dog', 'jumped', 'on', 'the', 'mat', '.']
coll1 = Collection([Doc({'token': line1}, 'cat-doc')])
coll2 = Collection([Doc({'token': line2}, 'dog-doc')])
```

`Collection.from_file` assumes that the input are files with a sentence per line (although it can also perform shingling on the input text).

`Collection.from_csv` uses a csv file (or more) as input. Typically, this file will have one `token`, a `lemma` and `pos` fields.

```
$ head input.csv

token   lemma   pos
The     the     DET
cat     cat     N
sat     sit     V
```

## Preprocessing

Preprocessing is done with the `TextPreprocessor` class. In order to lowercase the input and filter out punctuation and stopwords, we can use the following snippet. For this example, we use one of the built-in datasets that come prepackaged with RETRIEVE.

```python
>>> from retrieve.data import TextPreprocessor
>>> from retrieve.utils import Stopwords
>>> from retrieve.corpora import load_vulgate

>>> coll = load_vulgate()
>>> TextPreprocessor(
        stopwords=Stopwords('latin.stop'), lower=True, drop_punctuation=True
>>> ).process_collections(coll)

>>> coll[0].get_features()
['principium', 'creo', 'deus', 'caelum', 'terra']
```

We can also compute n-grams using the `min_n` and `max_n` arguments.

```python
>>> coll = load_vulgate()
>>> TextPreprocessor(
        stopwords=Stopwords('latin.stop'), lower=True, drop_punctuation=True
>>> ).process_collections(coll, min_n=1, max_n=3)

>>> coll[0].get_features()
['principium',
 'creo',
 'deus',
 'caelum',
 'terra',
 'principium--creo',
 'creo--deus',
 'deus--caelum',
 'caelum--terra',
 'principium--creo--deus',
 'creo--deus--caelum',
 'deus--caelum--terra']
```

**Feature selection** can be done using the `retrieve.data.FeatureSelector` class, in combination with the `retrieve.data.Criterion` class. We can do feature selection based on:
- document frequency (using `Criterion.DF`)
- raw frequency (using `Criterion.FREQ`)
- inverse document frequency (using `Criterion.IDF`)

For example, in order to filter out features that occur in only isolated documents, we use the following code.

```python
>>> vocab = FeatureSelector(coll).filter_collections(coll, criterion=(Criterion.DF >= 2))
```

`FeatureSelector.filter_collections` returns the vocabulary of features after filtering.

`Criterion` can be combined using ordinary operators. For example, `(Criterion.DF >= 2) & (Criterion.FREQ >= 5)` drops hapaxes and features with less than 5 occurrences overall.

## Search

RETRIEVE implements three algorithm families. 
- Set-based (Inverted-list approaches to efficient set-similarity measures)
- VSM-based (Vector Space Models including an optimized implementation of the soft-cosine measure)
- Local text alignment (Smith-Waterman)

### Set-based

### VSM

### Text-Alignment

# Quickrun

For simplicity, all functionality has been packed into a single `pipeline` function.

# Visualization


