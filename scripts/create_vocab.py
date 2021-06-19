import io
import os
import re
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import plac
from spacy.language import Language
from spacy.util import get_lang_class

sys.path.insert(0, os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir))))

from preturnie.pipeline.preturn_tokenizer import PreturnTokenizer  # pylint: disable=wrong-import-position
from preturnie.data_utils.filter_data import clean_data


def normalize_note_text(note):
    note = re.sub(r'[\r\n]', ' ', note)
    note = re.sub(r'^M', ' ', note)
    note = re.sub(r'\[\_.+\_\]', ' ', note)
    return note


def count_frequencies(language_class: Language, input_corpus: list):
    """
    Given a file containing single documents per line
    (for scispacy, these are Pubmed abstracts), split the text
    using a science specific tokenizer and compute word and
    document frequencies for all words.
    """
    print(f"Processing {input_corpus}.")
    tokenizer = PreturnTokenizer(language_class()).get_tokenizer()
    counts = Counter()
    doc_counts = Counter()
    for line in input_corpus:
        line = normalize_note_text(line)
        words = [t.text for t in tokenizer(line)]
        counts.update(words)
        doc_counts.update(set(words))

    return counts, doc_counts


def parallelize(func, iterator, n_jobs):
    pool = Pool(processes=n_jobs)
    counts = pool.starmap(func, iterator)
    return counts


def merge_counts(frequencies: List[Tuple[Counter, Counter]], output_path: str):
    """
    Merge a number of frequency counts generated from `count_frequencies`
    into a single file, written to `output_path`.
    """
    counts = Counter()
    doc_counts = Counter()
    for word_count, doc_count in frequencies:
        counts.update(word_count)
        doc_counts.update(doc_count)
    with io.open(output_path, 'w+', encoding='utf8') as file_:
        for word, count in counts.most_common():
            if not word.isspace():
                file_.write(f"{count}\t{doc_counts[word]}\t{repr(word)}\n")


@plac.annotations(
        input_file=("Location of input file", "positional", None, Path),
        output_file=("Location for output file", "positional", None, Path),
        n_jobs=("Number of workers", "option", "n", int))
def main(input_file: Path, output_file: Path, n_jobs=2):

    lang_class = get_lang_class('nl')
    nlp = lang_class()
    preturn_tokenizer = PreturnTokenizer(nlp).get_tokenizer()
    nlp.tokenizer = preturn_tokenizer

    df = pd.read_csv(input_file, encoding='utf-8')
    df = clean_data(df)

    notes = df['value_text'].values
    print(len(notes))
    counts = [count_frequencies(lang_class, notes)]
    print("Merge")
    merge_counts(counts, output_file)


if __name__ == '__main__':
    plac.call(main)
