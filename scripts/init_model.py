# coding: utf8

import gzip
import json
import math
import os
import sys
import tarfile
import zipfile
from ast import literal_eval

import numpy
import plac
import spacy
from preshed.counter import PreshCounter
from scispacy.file_cache import cached_path
from spacy.util import ensure_path
from spacy.vectors import Vectors
from tqdm import tqdm
from wasabi import Printer

from preturnie.pipeline.preturn_tokenizer import PreturnTokenizer

msg = Printer()

VERSION = 0.1


@plac.annotations(
        lang=("model language", "positional", None, str),
        output_dir=("model output directory", "positional", None, str),
        freqs_loc=("location of words frequencies file", "positional", None, str),
        vectors_loc=("optional: location of vectors file in Word2Vec format "
                     "(either as .txt or zipped as .zip or .tar.gz)", "option",
                     "v", str),
        no_expand_vectors=("optional: Whether to expand vocab with words found in vector file",
                           "flag", "x", bool),
        meta_overrides=("optional: meta_json file to load.",
                        "option", "m", str),
        prune_vectors=("optional: number of vectors to prune to",
                       "option", "V", int),
        min_word_frequency=("optional: Word frequency to prune vocab to.",
                       "option", "mwf", int)
)
def init_model(lang, output_dir, freqs_loc=None,
               vectors_loc=None, no_expand_vectors=False,
               meta_overrides=None, prune_vectors=-1, min_word_frequency=1):
    """
    Create a new model from raw data, like word frequencies, Brown clusters
    and word vectors.
    """
    output_dir = ensure_path(output_dir)
    if vectors_loc is not None:
        vectors_loc = cached_path(vectors_loc)
    if freqs_loc is not None:
        freqs_loc = cached_path(freqs_loc)
    freqs_loc = ensure_path(freqs_loc)

    if freqs_loc is not None and not freqs_loc.exists():
        msg.fail("Can't find words frequencies file", freqs_loc, exits=1)
    probs, oov_prob = read_freqs(freqs_loc, min_freq=min_word_frequency) if freqs_loc is not None else ({}, -20)

    vectors_data = None
    vector_keys = None
    if vectors_loc is not None:
        vectors_loc = ensure_path(vectors_loc)
        vectors_data, vector_keys = read_vectors(vectors_loc) if vectors_loc else (None, None)

    nlp = create_model(lang, probs, oov_prob, vectors_data, vector_keys, not no_expand_vectors, prune_vectors)
    nlp.tokenizer = PreturnTokenizer(nlp).get_tokenizer()
    ner = nlp.create_pipe('ner')
    nlp.add_pipe(ner)
    nlp.add_pipe(nlp.create_pipe('sentencizer'))
    # Insert our custom tokenizer into the base model.
    nlp.begin_training()

    if meta_overrides is not None:
        metadata = json.load(open(meta_overrides))
        nlp.meta.update(metadata)
        nlp.meta["version"] = VERSION

    if not output_dir.exists():
        os.makedirs(output_dir, exist_ok=True)

    nlp.to_disk(output_dir)
    return nlp


def open_file(loc):
    '''Handle .gz, .tar.gz or unzipped files'''
    loc = ensure_path(loc)
    print("Open loc")
    if tarfile.is_tarfile(str(loc)):
        return tarfile.open(str(loc), 'r:gz')
    elif loc.parts[-1].endswith('gz'):
        return (line.decode('utf8') for line in gzip.open(str(loc), 'r'))
    elif loc.parts[-1].endswith('zip'):
        zip_file = zipfile.ZipFile(str(loc))
        names = zip_file.namelist()
        file_ = zip_file.open(names[0])
        return (line.decode('utf8') for line in file_)
    else:
        return loc.open('r', encoding='utf8')


def create_model(lang, probs, oov_prob, vectors_data, vector_keys, expand_vectors, prune_vectors):
    print("Creating model...")
    nlp = spacy.blank(lang)

    for lexeme in nlp.vocab:
        lexeme.rank = 0
    lex_added = 0
    for i, (word, prob) in enumerate(tqdm(sorted(probs.items(), key=lambda item: item[1], reverse=True))):
        lexeme = nlp.vocab[word]
        lexeme.rank = i
        lexeme.prob = prob
        # Decode as a little-endian string, so that we can do & 15 to get
        # the first 4 bits.  _parse_features.pyx
        lexeme.cluster = 0
        lex_added += 1

    nlp.vocab.cfg.update({'oov_prob': oov_prob})
    if vector_keys is not None:
        new_keys = []
        new_indices = []
        for i, word in enumerate(vector_keys):
            if word not in nlp.vocab and expand_vectors:
                lexeme = nlp.vocab[word]
                lex_added += 1
            elif word in nlp.vocab and not expand_vectors:
                new_keys.append(word)
                new_indices.append(i)
        print('Vector keys not none')

        if len(vectors_data):
            if not expand_vectors:
                print("New vectors")
                nlp.vocab.vectors = Vectors(data=vectors_data, keys=vector_keys)
            else:
                print("Expanding vectors")
                nlp.vocab.vectors = Vectors(data=vectors_data[new_indices], keys=new_keys)

        if prune_vectors >= 1:
            nlp.vocab.prune_vectors(prune_vectors)

    vec_added = len(nlp.vocab.vectors)
    msg.good(
        "Sucessfully compiled vocab",
        "{} entries, {} vectors".format(lex_added, vec_added),
    )

    return nlp


def read_vectors(vectors_loc):
    print("Reading vectors from %s" % vectors_loc)
    f = open_file(vectors_loc)
    shape = tuple(int(size) for size in next(f).split())
    vectors_data = numpy.zeros(shape=shape, dtype='f')
    vectors_keys = []
    for i, line in enumerate(tqdm(f)):
        line = line.rstrip()
        pieces = line.rsplit(' ', vectors_data.shape[1]+1)
        word = pieces.pop(0)
        if len(pieces) != vectors_data.shape[1]:
            # raise ValueError(Errors.E094.format(line_num=i, loc=vectors_loc))
            vectors_keys.append('UNK')
            continue
        else:
            vectors_data[i] = numpy.asarray(pieces, dtype='f')
            vectors_keys.append(word)
    return vectors_data, vectors_keys


def read_freqs(freqs_loc, max_length=100, min_doc_freq=5, min_freq=50):
    print("Counting frequencies...")
    counts = PreshCounter()
    total = 0
    with freqs_loc.open() as f:
        for i, line in enumerate(f):
            freq, doc_freq, key = line.rstrip().split('\t', 2)
            freq = int(freq)
            counts.inc(i + 1, freq)
            total += freq
    counts.smooth()
    log_total = math.log(total)
    probs = {}
    with freqs_loc.open() as f:
        for line in tqdm(f):
            freq, doc_freq, key = line.rstrip().split('\t', 2)
            doc_freq = int(doc_freq)
            freq = int(freq)
            if doc_freq >= min_doc_freq and freq >= min_freq and len(key) < max_length:
                try:
                    word = literal_eval(key)
                except SyntaxError:
                    # Take odd strings literally.
                    word = literal_eval("'%s'" % key)
                smooth_count = counts.smoother(int(freq))
                probs[word] = math.log(smooth_count) - log_total
    oov_prob = math.log(counts.smoother(0)) - log_total
    return probs, oov_prob


if __name__ == '__main__':
    plac.call(init_model)
