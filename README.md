
This repository contains the information extraction pipeline used to extract features from semi-structured and structured from medical notes related to preterm birth.
Extracted features were used to support clinical decision making in the context of preterm birth risk prediction.
This library was developed for the B-project preturn at [Gent University Hospital](https://www.uzgent.be).


This pipeline is largely based on the spacy and [scispacy](https://github.com/allenai/scispacy) libraries.


## Data

 Because of its sensitive nature, for now, no raw text and training data is included.


## Installation

To use the repo create a virtual environment and call

    python3 -m venv .env
    source .env/bin/activate
    pip install .

Generate embeddings and vocabulary

    python -m scripts.generate_embeddings --embedding_size 300 --window_size 2
    python -m scripts.create_vocab data/input/corpus.csv data/input/counts.freq

Convert UMLS database to json

    python -m scripts.export_umls_json --meta_path data/umls/umls/ --output_path data/umls/umls_2017_aa_cat0129.json

Generate UMLS inverted index vectors

    python -m scripts.train_linker --umls_path data/umls/umls_2017_aa_cat0129 --train

## Train weakly supervised NER

Weak training data is created by labeling unlabeled text using patterns stored in data/input/patterns/entities.json

    python -m scripts.extract_features

This stores the weakly supervised training data in files data/weak_supervision/{training,dev}_data.json
A weakly supervised NER model can then be trained calling

    python -m scripts.train_ner nl ./ data/input/training_data.json data/input/dev_data.json -b models/preturn


## Example

To extract features from medical notes in data/input/corpus.csv and output a csv containing extracted features, call

    python -m scripts.extract_features

```python

from preturn_ie import load_pipeline

text_doc = ("2 maal 1000 mg Dafalgan")

nlp = load_preturn_model(model_path='models/preturn')
doc = nlp(text_doc)

for feature in doc._.features:
    pprint(feature)

''' 
    {
        'attribute': '',
        'canonical_name': '',
        'concept_id': ' ',
        'drug_name': 'dafalgan',
        'feature_name': 'DRUG_ADMINISTRATION',
        'feature_string': '_2_keer_1000_mg_dafalgan',
        'feature_type': 'drug',
        'match_id': '',
        'modifier': '',
        'source_text': '2 keer 1000 mg dafalgan',
        'unit_name': ' TIMES  MASS_UNIT MEDICATION',
        'value': 2.0
    }
 '''
```

## 




## Citations and Acknowledgments

Should you use this code for your own research, please cite:

```
@article{STERCKX2020103544,
    title = "Clinical information extraction for preterm birth risk prediction",
    journal = "Journal of Biomedical Informatics",
    volume = "110",
    pages = "103544",
    year = "2020",
    issn = "1532-0464",
    doi = "https://doi.org/10.1016/j.jbi.2020.103544",
    url = "http://www.sciencedirect.com/science/article/pii/S1532046420301726",
    author = "Lucas Sterckx and Gilles Vandewiele and Isabelle Dehaene and Olivier Janssens and Femke Ongenae and Femke {De Backere} and Filip {De Turck} and Kristien Roelens and Johan Decruyenaere and Sofie {Van Hoecke} and Thomas Demeester",
    keywords = "Clinical information extraction, Clinical decision support models, Preterm birth, Text mining"
}
```
