import json
import multiprocessing
import operator
from spacy.util import get_lang_class
from collections import defaultdict
import collections

import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import logging
import re
import argparse

from preturnie.pipeline.preturn_tokenizer import PreturnTokenizer
from preturnie.data_utils.filter_data import clean_data


def normalize_note_text(note):
    note = re.sub(r'[\r\n]', ' ', note)
    note = re.sub(r'^M', ' ', note)
    note = re.sub(r'\[\_.+\_\]', ' ', note)
    note = note.replace("[bod or phone]", "bod_or_phone")
    note = re.sub(r'_given_name_\d+_', '_given_name_', note)
    note = re.sub(r'_doctor_\d+_', '_doctor_', note)
    return note


def extract_sentences_from_notes(notes):
    lang_class = get_lang_class('nl')
    nlp = lang_class()
    preturn_tokenizer = PreturnTokenizer(nlp).get_tokenizer()
    nlp.tokenizer = preturn_tokenizer

    sentences = [normalize_note_text(note) for note in notes]
    sentences = [[t.text for t in nlp(note)] for note in sentences]
    return sentences


def token_count(sentences):
    token_count = defaultdict(int)

    for sent in sentences:
        for token in sent:
            token_count[token] += 1

    print('Unique tokens:', len(token_count))
    print()

    token_count = sorted(token_count.items(), key=operator.itemgetter(1), reverse=True)

    print('Top tokens:', token_count[:25])
    print()


def train_w2v_model(sentences, embedding_size, window_size):
    w2v_model = Word2Vec(
        min_count=1,
        size=embedding_size,
        window=window_size,
        sample=1e-4,
        negative=2,
        workers=multiprocessing.cpu_count()
    )

    w2v_model.build_vocab(sentences)
    w2v_model.train(sentences, total_examples=w2v_model.corpus_count, epochs=100, report_delay=1)

    print('Unique tokens with at least 2 occurrences:', len(w2v_model.wv.vocab))
    print()

    return w2v_model


def check_similar_tokens(model):

    sorted_vocab = []
    pattern_files = ['data/patterns/entities.json']
    for pattern_file in pattern_files:
        patterns = json.load(open(pattern_file, 'r'))
        for entry in patterns:
            if 'regex' not in str(entry).lower():
                label = entry["label"]
                text = ' '.join([list(pattern_token.values())[0] for pattern_token in entry['pattern']]).lower().split()[0]
                sorted_vocab.append([label, text])

    with open('reports/w2v_output.txt', 'w') as fp:
        for label, token in list(sorted_vocab)[:1000]:
            if token in list(w2v_model.wv.vocab):
                fp.write(str(token)+'\t\n')
                for i, similar_token in enumerate(w2v_model.wv.most_similar(positive=[token])):
                    fp.write(str(i)+'\t' + str(similar_token)+'\n')
                    if similar_token[0] not in [voc_token[1] for voc_token in sorted_vocab]:
                        print('{"label":"'+label+'","pattern":  [{"TEXT": "' + similar_token[0] + '"}]},')

    plot_embeddings(model, sorted_vocab)


def save_model_and_embeddings(w2v_model):
    w2v_model.save('data/embeddings/w2v.model')
    w2v_model.wv.save_word2vec_format('data/embeddings/w2v_embeddings.txt')


def plot_embeddings(model, sorted_vocab):
    tokens = []
    vectors = []

    for label, word in list(sorted_vocab)[:2000]:
        if word in list(model.wv.vocab):
            vectors.append(model.wv[word])
            tokens.append(word)

    tsne = TSNE(n_components=2, random_state=1)
    vectors_transformed = tsne.fit_transform(vectors)

    df_tsne = pd.DataFrame(vectors_transformed,
                           index=tokens, columns=['x', 'y'])
    fig = plt.gcf()
    fig.set_size_inches(18.5, 10.5)
    plt.scatter(df_tsne['x'], df_tsne['y'])

    for token, xy in df_tsne.iterrows():
        plt.annotate(token, xy)

    plt.savefig('reports/figs/tsne.png')


def output_freq_vocab(text):

    def ngrams(subtext, n=2):
        return zip(*[subtext[i:] for i in range(n)])

    with open('data/input/freq_vocab.txt', 'w') as fp:

        ngram_counts = collections.Counter()
        for i in range(1, 5):
            for line in text:
                ngram_counts.update(ngrams(line, n=i))

        print(ngram_counts.most_common(10))
        for token, count in list(ngram_counts.most_common(200000)):
            fp.write(str(' '.join(token)) + '\n')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--embedding_size',
            default=300,
            help='Size of word embedding vectors'
    )
    parser.add_argument(
            '--window_size',
            default=2,
            help='Window size for skipgram training'
    )
    args = parser.parse_args()

    df = pd.read_csv('data/input/corpus.csv')
    df = clean_data(df)

    notes = df['value_text'].values
    sentences = extract_sentences_from_notes(notes)
    output_freq_vocab(sentences)

    embedding_size = int(args.embedding_size)
    window_size = int(args.window_size)

    w2v_model = train_w2v_model(sentences, embedding_size, window_size)
    save_model_and_embeddings(w2v_model)
    check_similar_tokens(w2v_model)
