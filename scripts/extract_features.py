
import json
from collections import Counter

import pandas as pd
from spacy.gold import docs_to_json
from preturnie.data_utils.filter_data import clean_data
from preturnie.load_pipeline import load_preturn_model


DEV_SIZE = 500


def extract_features(df, note_type=[], nrows=100, extract_features=True, enable_linker=True, prefix=''):

    if note_type:
        print(f'Parsing notes of type \'{note_type}\'')
        print('===')
        notes = df[df['templ_text'].isin(list(note_type))]
        df[df['templ_text'].isin(list(note_type))].to_csv('data/input/{}notes.csv'.format(prefix))
    else:
        print('Parsing all notes')
        notes = df
        df.to_csv('data/input/{}notes.csv'.format(prefix))

    notes['text'] = notes['value_text']
    notes.to_json('data/input/{}notes.jsonl'.format(prefix), orient='records', lines=True)
    amount = min(nrows, len(notes))

    nlp = load_preturn_model(model_path="models/preturn")

    structured_notes = []
    parsed_docs = []
    train_data = []
    entities = {}
    umls_entities = {}

    for i in range(amount):

        text_doc = notes.iloc[i]['value_text']
        doc = nlp(text_doc)

        if i < 200 or i % 300 == 0:
            if len(doc.ents) > 0:
                parsed_docs.append(doc)

            print("Parsing Note {}:".format(i))
            print("Text:")
            print(notes.iloc[i]['value_text'])

            for ent in doc.ents:
                print("Entities:")
                print(ent.text, ent.start_char, ent.end_char, ent.label_, ent._.value)
                if ent.label_ not in entities:
                    entities[ent.label_] = Counter()
                entities[ent.label_].update([ent.text.lower()])

                if enable_linker and extract_features:
                    if ent._.umls_ents:
                        print("Linked Entities:")
                        print(nlp.pipeline[-2][1].umls.cui_to_entity[ent._.umls_ents[0][0]])
                        if ent.label_ not in umls_entities:
                            umls_entities[ent.label_] = Counter()
                        umls_entities[ent.label_].update([str(nlp.pipeline[-2][1].umls.cui_to_entity[ent._.umls_ents[0][0]]).split('\n')[0]])

        if extract_features:
            for feature in doc._.features:
                record = feature
                record['pregnancy_key'] = notes.iloc[i]['pregnancy_key']
                record['note_index'] = notes.iloc[i]['note_index']
                record['preturn_id'] = notes.iloc[i]['preturn_id']
                record['admission_index'] = notes.iloc[i]['admission_index']
                record['creation_date'] = notes.iloc[i]['creation_date']
                record['templ_text'] = notes.iloc[i]['templ_text']
                record['seconds_since_admission'] = notes.iloc[i]['seconds_since_admission']
                structured_notes.append(record)

        if len(doc.ents) > 0:
            train_data.append(doc)

    train_json = docs_to_json(train_data[:-1*DEV_SIZE])
    with open('data/weak_supervision/{}training_data.json'.format(prefix), 'w') as f:
        json.dump([train_json], f)
    dev_json = docs_to_json(train_data[-1*DEV_SIZE:])
    with open('data/weak_supervision/{}dev_data.json'.format(prefix), 'w') as f:
        json.dump([dev_json], f)

    structured_df = pd.DataFrame(structured_notes)
    structured_df['creation_date'] = pd.to_datetime(structured_df['creation_date'])
    structured_df['value'] = structured_df['value'].astype('float')
    structured_df['preturn_id'] = structured_df['preturn_id'].astype('int')
    structured_df['pregnancy_key'] = structured_df['pregnancy_key'].astype('int')
    structured_df['seconds_since_admission'] = structured_df['seconds_since_admission'].astype('int')
    structured_df['note_index'] = structured_df['note_index'].astype('int')
    structured_df['admission_index'] = structured_df['admission_index'].astype('int')
    structured_df.to_csv('data/features/{}notes_features.csv'.format(prefix))


if __name__ == '__main__':

    df = pd.read_csv('data/input/corpus.csv')
    df = clean_data(df)
    extract_features(df, extract_features=True, enable_linker=True)
