
import spacy
import requests
import re

titles = [
    "Down the Rabbit-Hole",
    "The Pool of Tears",
    "A Caucus-Race and a Long Tale",
    "The Rabbit Sends in a Little Bill",
    "Advice from a Caterpillar",
    "Pig and Pepper",
    "A Mad Tea-Party",
    "The Queen's Croquet-Ground",
    "The Mock Turtle's Story",
    "The Lobster Quadrille",
    "Who Stole the Tarts?",
    "Alice's Evidence"]


def make_safe_filename(filename):
    return "".join([c for c in filename 
        if c in ['-', '_'] or c.isalpha() or c.isdigit() or c==' ']).rstrip()


if __name__ == '__main__':
    nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    nlp.tokenizer.infix_finditer = re.compile(r'''[-—~]''').finditer

    # Alice in Wonderland
    path = 'https://www.gutenberg.org/files/11/11-0.txt'
    res = requests.get(path)
    res.encoding = res.apparent_encoding
    text = res.text.strip()

    _, *chapters = re.split(r'CHAPTER [IVX]+\.\r\n', text)
    # drop legal notice
    chapters[-1] = re.split('THE END', chapters[-1])[0]

    assert len(titles) == len(chapters)

    for idx, chapter in enumerate(chapters):
        chapter = chapter.split('\r\n')
        chapter = ' '.join([line.strip() for line in chapter if line.strip()])
        # remove underlining like "_this_"
        chapter = re.sub(r'_([^_]+)_', '\\1', chapter)
        title = 'Chapter_{}_{}'.format(idx + 1, titles[idx])
        title = make_safe_filename(title)
        with open('src/retrieve/resources/texts/alice/' + title + '.txt', 'w') as f:
            # process with spacy
            for token in nlp(chapter):
                # avoid spaces
                if token.pos_ == 'SPACE':
                    continue
                f.write('{}\t{}\t{}\n'.format(token.text, token.lemma_, token.pos_))

