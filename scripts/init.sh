python -m scripts.generate_embeddings --embedding_size 300 --window_size 2
python -m scripts.create_vocab data/input/corpus.csv data/input/counts.freq
python -m scripts.export_umls_json --meta_path data/umls/umls/ --output_path data/umls/umls_2017_aa_cat0129.json
python -m scripts.train_linker --umls_path data/umls/umls_2017_aa_cat0129.json --train
python -m scripts.train_ner nl ./models/trained data/input/training_data.json data/input/training_data.json -b models/preturn
python -m scripts.extract_features