"""
Linking using char-n-gram with approximate nearest neighbors.
"""

import argparse
import datetime
import json
from collections import defaultdict
from typing import Any, Dict, List, NamedTuple, Set, Tuple

import nmslib
import numpy as np
import scipy
from joblib import dump, load
from nmslib.dist import FloatIndex
from scispacy import data_util
from sklearn.base import ClassifierMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.language import Language
from spacy.tokens import Span
from tqdm import tqdm


def load_umls_kb(umls_path: str) -> List[Dict]:
    """
    Reads a UMLS json release and return it as a list of concepts.
    Each concept is a dictionary.
    """
    with open(umls_path) as f:
        print(f'Loading umls concepts from {umls_path}')
        umls_concept_list = json.load(f)
    return umls_concept_list


class MentionCandidate(NamedTuple):
    concept_id: str
    distances: List[float]
    aliases: List[str]


class CandidateGenerator:

    """
    A candidate generator for entity linking to the Unified Medical Language System (UMLS).

    It uses a sklearn.TfidfVectorizer to embed mention text into a sparse embedding of character 3-grams.
    These are then compared via cosine distance in a pre-indexed approximate nearest neighbours index of
    a subset of all entities and aliases in UMLS.

    Once the K nearest neighbours have been retrieved, they are canonicalized to their UMLS canonical ids.
    This step is required because the index also includes entity aliases, which map to a particular canonical
    entity. This point is important for two reasons:

    1. K nearest neighbours will return a list of Y possible neighbours, where Y < K, because the entity ids
    are canonicalized.

    2. A single string may be an alias for multiple canonical entities. For example, "Jefferson County" may be an
    alias for both the canonical ids "Jefferson County, Iowa" and "Jefferson County, Texas". These are completely
    valid and important aliases to include, but it means that using the candidate generator to implement a naive
    k-nn baseline linker results in very poor performance, because there are multiple entities for some strings
    which have an exact char3-gram match, as these entities contain the same alias string. This situation results
    in multiple entities returned with a distance of 0.0, because they exactly match an alias, making a k-nn baseline
    effectively a random choice between these candidates. However, this doesn't matter if you have a classifier
    on top of the candidate generator, as is intended!

    Parameters
    ----------
    ann_index: FloatIndex
        An nmslib approximate nearest neighbours index.
    tfidf_vectorizer: TfidfVectorizer
        The vectorizer used to encode mentions.
    ann_concept_aliases_list: List[str]
        A list of strings, mapping the indices used in the ann_index to canonical UMLS ids.
    mention_to_concept: Dict[str, Set[str]], required.
        A mapping from aliases to canonical ids that they are aliases of.
    verbose: bool
        Setting to true will print extra information about the generated candidates

    """
    def __init__(self,
                 ann_index: FloatIndex,
                 tfidf_vectorizer: TfidfVectorizer,
                 ann_concept_aliases_list: List[str],
                 mention_to_concept: Dict[str, Set[str]],
                 verbose: bool = True) -> None:

        self.ann_index = ann_index
        self.vectorizer = tfidf_vectorizer
        self.ann_concept_aliases_list = ann_concept_aliases_list
        self.mention_to_concept = mention_to_concept
        self.verbose = verbose

    def nmslib_knn_with_zero_vectors(self, vectors: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        ann_index.knnQueryBatch crashes if any of the vectors is all zeros.
        This function is a wrapper around `ann_index.knnQueryBatch` that solves this problem. It works as follows:
        - remove empty vectors from `vectors`.
        - call `ann_index.knnQueryBatch` with the non-empty vectors only. This returns `neighbors`,
        a list of list of neighbors. `len(neighbors)` equals the length of the non-empty vectors.
        - extend the list `neighbors` with `None`s in place of empty vectors.
        - return the extended list of neighbors and distances.
        """
        empty_vectors_boolean_flags = np.array(vectors.sum(axis=1) != 0).reshape(-1,)
        empty_vectors_count = vectors.shape[0] - sum(empty_vectors_boolean_flags)
        if self.verbose:
            print(f'Number of empty vectors: {empty_vectors_count}')

        # remove empty vectors before calling `ann_index.knnQueryBatch`
        vectors = vectors[empty_vectors_boolean_flags]

        # call `knnQueryBatch` to get neighbors
        original_neighbours = self.ann_index.knnQueryBatch(vectors, k=k)
        neighbors, distances = zip(*[(x[0].tolist(), x[1].tolist()) for x in original_neighbours])
        neighbors = list(neighbors)
        distances = list(distances)
        # all an empty list in place for each empty vector to make sure len(extended_neighbors) == len(vectors)

        # init extended_neighbors with a list of Nones
        extended_neighbors = np.empty((len(empty_vectors_boolean_flags),), dtype=object)
        extended_distances = np.empty((len(empty_vectors_boolean_flags),), dtype=object)

        # neighbors need to be convected to an np.array of objects instead of ndarray of dimensions len(vectors)xk
        # Solution: add a row to `neighbors` with any length other than k. This way, calling np.array(neighbors)
        # returns an np.array of objects
        neighbors.append([])
        distances.append([])
        # interleave `neighbors` and Nones in `extended_neighbors`
        extended_neighbors[empty_vectors_boolean_flags] = np.array(neighbors)[:-1]
        extended_distances[empty_vectors_boolean_flags] = np.array(distances)[:-1]

        return extended_neighbors, extended_distances

    def generate_candidates(self, mention_texts: List[str], k: int) -> List[Dict[str, List[int]]]:
        """
        Given a list of mention texts, returns a list of candidate neighbors.

        NOTE: Because we include canonical name aliases in the ann index, the list
        of candidates returned will not necessarily be of length k for each candidate,
        because we then map these to canonical ids only.
        # TODO Mark: We should be able to use this signal somehow, maybe a voting system?
        args:
            mention_texts: list of mention texts
            k: number of ann neighbors

        returns:
            A list of dictionaries, each containing the mapping from umls concept ids -> a list of
            the cosine distances between them. Note that these are lists for each concept id, because
            the index contains aliases which are canonicalized, so multiple values may map to the same
            canonical id.
        """
        if self.verbose:
            print(f'Generating candidates for {len(mention_texts)} mentions')
        tfidfs = self.vectorizer.transform(mention_texts)
        start_time = datetime.datetime.now()

        # `ann_index.knnQueryBatch` crashes if one of the vectors is all zeros.
        # `nmslib_knn_with_zero_vectors` is a wrapper around `ann_index.knnQueryBatch` that addresses this issue.
        batch_neighbors, batch_distances = self.nmslib_knn_with_zero_vectors(tfidfs, k)
        end_time = datetime.datetime.now()
        total_time = end_time - start_time
        if self.verbose:
            print(f'Finding neighbors took {total_time.total_seconds()} seconds')
        neighbors_by_concept_ids = []
        for neighbors, distances in zip(batch_neighbors, batch_distances):
            if neighbors is None:
                neighbors = []

            if distances is None:
                distances = []
            predicted_umls_concept_ids = defaultdict(list)
            for n, d in zip(neighbors, distances):
                mention = self.ann_concept_aliases_list[n]
                concepts_for_mention = self.mention_to_concept[mention]
                for concept_id in concepts_for_mention:
                    predicted_umls_concept_ids[concept_id].append((mention, d))

            neighbors_by_concept_ids.append({**predicted_umls_concept_ids})
        return neighbors_by_concept_ids


class Linker:
    """
    An entity linker for the Unified Medical Language System (UMLS).

    Given a mention and a list of candidates (generated by CandidateGenerator), it uses an sklearn classifier
    to sort the candidates by their probabilty of being the the right entity for the mention.

    Parameters
    ----------
    umls_concept_dict_by_id: Dict
        A dictionary of the UMLS concepts.
    classifier: ClassifierMixin
        An sklearn classifier that takes a mention and a candidate and evaluate them.
        If classifier is None, the linking function returns the same list with no sorting.
        Also, `classifier_example` and `featurizer` functions are still useful for generating
        classifier training data.
    """

    def __init__(self,
                 umls_concept_dict_by_id: Dict,
                 classifier: ClassifierMixin = None) -> None:
        self.umls_concept_dict_by_id = umls_concept_dict_by_id
        self.classifier = classifier

    @classmethod
    def featurizer(cls, example: Dict):
        """Featurize a dictionary of values for the linking classifier."""
        features = []
        features.append(int(example['has_definition']))  # 0 if candidate doesn't have definition, 1 otherwise

        features.append(min(example['distances']))
        features.append(max(example['distances']))
        features.append(len(example['distances']))
        features.append(np.mean(example['distances']))

        gold_types = set(example['mention_types'])
        candidate_types = set(example['candidate_types'])

        features.append(len(gold_types))
        features.append(len(candidate_types))
        features.append(len(candidate_types.intersection(gold_types)))

        return features

    def classifier_example(self, candidate_id: str, candidate: List[Tuple[str, float]], mention_text: str, mention_types: List[str]):
        """Given a candidate and a mention, return a dictionary summarizing relevant information for classification."""
        has_definition = 'definition' in self.umls_concept_dict_by_id[candidate_id]
        distances = [distance for aliase, distance in candidate]
        candidate_types = self.umls_concept_dict_by_id[candidate_id]['types']

        return {'has_definition': has_definition,
                'distances': distances,
                'mention_types': mention_types,
                'candidate_types': candidate_types}

    def link(self, candidates: Dict[str, List[Tuple[str, float]]], mention_text: str, mention_types: List[str]):
        """
        Given a dictionary of candidates and a mention, return a list of candidate ids sorted by
        probability it is the right entity for the mention.

        args:
            candidates: dictionary of candidates of the form candidate id -> list((aliase, distance)).
            mention_text: mention text.
            mention_types: list of mention types.

        returns:
            A list of candidate ids sorted by the probability it is the right entity for the mention.
        """
        features = []
        candidate_ids = list(candidates.keys())
        if self.classifier is None:
            return candidate_ids

        for candidate_id in candidate_ids:
            candidate = candidates[candidate_id]
            classifier_example = self.classifier_example(candidate_id, candidate, mention_text, mention_types)
            features.append(self.featurizer(classifier_example))
        if len(features) == 0:
            return []
        scores = self.classifier.predict_proba(features)
        return [candidate_ids[i] for i in np.argsort(-scores[:, 1], kind='mergesort')]  # mergesort is stable


def create_tfidf_ann_index(model_path: str, text_to_concept: Dict[str, Set[str]]) -> None:
    """
    Build tfidf vectorizer and ann index.
    """
    tfidf_vectorizer_path = f'{model_path}/tfidf_vectorizer.joblib'
    ann_index_path = f'{model_path}/nmslib_index.bin'
    tfidf_vectors_path = f'{model_path}/tfidf_vectors_sparse.npz'
    uml_concept_aliases_path = f'{model_path}/concept_aliases.json'

    # nmslib hyperparameters (very important)
    # guide: https://github.com/nmslib/nmslib/blob/master/python_bindings/parameters.md
    # default values resulted in very low recall
    M = 100  # set to the maximum recommended value. Improves recall at the expense of longer indexing time
    efC = 2000  # `C` for Construction. Set to the maximum recommended value
                # Improves recall at the expense of longer indexing time
    efS = 1000  # `S` for Search. This controls performance at query time. Maximum recommended value is 2000.
                # It makes the query slow without significant gain in recall.
    num_threads = 60  # set based on the machine
    index_params = {'M': M, 'indexThreadQty': num_threads, 'efConstruction': efC, 'post': 0}

    print(f'No tfidf vectorizer on {tfidf_vectorizer_path} or ann index on {ann_index_path}')
    uml_concept_aliases = list(text_to_concept.keys())

    uml_concept_aliases = np.array(uml_concept_aliases)

    print(f'Fitting tfidf vectorizer on {len(uml_concept_aliases)} aliases')
    tfidf_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), min_df=2, dtype=np.float32)
    # tfidf_vectorizer = CountVectorizer(ngram_range=(3, 3), min_df=2, dtype=np.float32)

    start_time = datetime.datetime.now()
    uml_concept_alias_tfidfs = tfidf_vectorizer.fit_transform(uml_concept_aliases)
    print(f'Saving tfidf vectorizer to {tfidf_vectorizer_path}')
    dump(tfidf_vectorizer, tfidf_vectorizer_path)
    end_time = datetime.datetime.now()
    total_time = (end_time - start_time)
    print(f'Fitting and saving vectorizer took {total_time.total_seconds()} seconds')

    print(f'Finding empty (all zeros) tfidf vectors')
    empty_tfidfs_boolean_flags = np.array(uml_concept_alias_tfidfs.sum(axis=1) != 0).reshape(-1,)
    deleted_aliases = uml_concept_aliases[empty_tfidfs_boolean_flags == False]
    number_of_non_empty_tfidfs = len(deleted_aliases)
    total_number_of_tfidfs = uml_concept_alias_tfidfs.shape[0]

    print(f'Deleting {number_of_non_empty_tfidfs}/{total_number_of_tfidfs} aliases because their tfidf is empty')
    # remove empty tfidf vectors, otherwise nmslib will crash
    uml_concept_aliases = uml_concept_aliases[empty_tfidfs_boolean_flags]
    uml_concept_alias_tfidfs = uml_concept_alias_tfidfs[empty_tfidfs_boolean_flags]
    print(deleted_aliases)

    print(f'Saving list of concept ids and tfidfs vectors to {uml_concept_aliases_path} and {tfidf_vectors_path}')
    json.dump(uml_concept_aliases.tolist(), open(uml_concept_aliases_path, "w"))
    scipy.sparse.save_npz(tfidf_vectors_path, uml_concept_alias_tfidfs.astype(np.float16))

    print(f'Fitting ann index on {len(uml_concept_aliases)} aliases (takes 2 hours)')
    start_time = datetime.datetime.now()
    ann_index = nmslib.init(method='hnsw', space='cosinesimil_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)
    ann_index.addDataPointBatch(uml_concept_alias_tfidfs)
    ann_index.createIndex(index_params, print_progress=True)
    ann_index.saveIndex(ann_index_path)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f'Fitting ann index took {elapsed_time.total_seconds()} seconds')


def load_tfidf_ann_index(model_path: str):
    # `S` for Search. This controls performance at query time. Maximum recommended value is 2000.
    # It makes the query slow without significant gain in recall.
    efS = 1000

    tfidf_vectorizer_path = f'{model_path}/tfidf_vectorizer.joblib'
    ann_index_path = f'{model_path}/nmslib_index.bin'
    tfidf_vectors_path = f'{model_path}/tfidf_vectors_sparse.npz'
    uml_concept_aliases_path = f'{model_path}/concept_aliases.json'

    start_time = datetime.datetime.now()
    print(f'Loading list of concepted ids from {uml_concept_aliases_path}')
    uml_concept_aliases = json.load(open(uml_concept_aliases_path))

    print(f'Loading tfidf vectorizer from {tfidf_vectorizer_path}')
    tfidf_vectorizer = load(tfidf_vectorizer_path)
    if isinstance(tfidf_vectorizer, TfidfVectorizer):
        print(f'Tfidf vocab size: {len(tfidf_vectorizer.vocabulary_)}')

    print(f'Loading tfidf vectors from {tfidf_vectors_path}')
    uml_concept_alias_tfidfs = scipy.sparse.load_npz(tfidf_vectors_path).astype(np.float32)

    print(f'Loading ann index from {ann_index_path}')
    ann_index = nmslib.init(method='hnsw', space='cosinesimil_sparse', data_type=nmslib.DataType.SPARSE_VECTOR)
    ann_index.addDataPointBatch(uml_concept_alias_tfidfs)
    ann_index.loadIndex(ann_index_path)
    query_time_params = {'efSearch': efS}
    ann_index.setQueryTimeParams(query_time_params)

    end_time = datetime.datetime.now()
    total_time = (end_time - start_time)

    print(f'Loading concept ids, vectorizer, tfidf vectors and ann index took {total_time.total_seconds()} seconds')
    return uml_concept_aliases, tfidf_vectorizer, ann_index


def load_linking_classifier(model_path: str):
    linking_classifier_path = f'{model_path}/linking_classifier.joblib'

    print(f'Loading linking classifier from {linking_classifier_path}')
    try:
        linking_classifier = load(linking_classifier_path)
    except:
        print('Loading linking classifier failed')
        return None
    return linking_classifier


def get_mention_text_and_ids(data: List[data_util.MedMentionExample],
                             umls: Dict[str, Any]):
    missing_entity_ids = []  # entities in MedMentions but not in UMLS

    # don't care about context for now. Just do the processing based on mention text only
    # collect all the data in one list to use ann.knnQueryBatch which is a lot faster than
    # calling ann.knnQuery for each individual example
    mention_texts = []
    gold_umls_ids = []

    for example in data:
        for entity in example.entities:
            if entity.umls_id not in umls:
                missing_entity_ids.append(entity)  # the UMLS release doesn't contan all UMLS concepts
                continue

            mention_texts.append(entity.mention_text)
            gold_umls_ids.append(entity.umls_id)
            continue

    return mention_texts, gold_umls_ids, missing_entity_ids



def maybe_substitute_span(doc, entity, abbreviations):
    maybe_ent_span = doc.char_span(entity.start, entity.end)
    if maybe_ent_span is None:
        maybe_ent_span = doc.char_span(entity.start, entity.end + 1)
    if maybe_ent_span in abbreviations:
        return maybe_ent_span, str(abbreviations[maybe_ent_span])
    else:
        return None, None

def get_mention_text_and_ids_by_doc(data: List[data_util.MedMentionExample],
                                    umls: Dict[str, Any],
                                    nlp: Language,
                                    substitute_abbreviations=False):
    """
    Returns a list of tuples containing a MedMentionExample and the texts and ids contianed in it

    Parameters
    ----------
    data: List[data_util.MedMentionExample]
        A list of MedMentionExamples being evaluated
    umls: Dict[str, Any]
        A dictionary of UMLS concepts
    nlp : Language
        A spacy NLP model.
    substitute_abbreviations: bool, default = False
        Whether or not to search for and replace abbreviations when generating mention candidates.
        Note that this can be applied on both gold and predicted mentions.
    """
    missing_entity_ids = []  # entities in MedMentions but not in UMLS
    examples_with_labels = []

    substituted = 0
    total = 0

    for example in data:

        doc = nlp(example.text)
        abbreviations = {}
        if substitute_abbreviations:
            for short in doc._.abbreviations:
                abbreviations[short] = short._.long_form

        mention_texts = []
        predicted_mention_texts = []
        gold_umls_ids = []
        for entity in example.entities:
            if entity.umls_id not in umls:
                missing_entity_ids.append(entity)  # the UMLS release doesn't contan all UMLS concepts
                continue

            _, mention_string = maybe_substitute_span(doc, entity, abbreviations)
            if mention_string is None:
                mention_string = entity.mention_text

            if mention_string != entity.mention_text:
                substituted += 1
            mention_texts.append(mention_string)

            gold_umls_ids.append(entity.umls_id)
            total += 1


        # Note that because we might substitute some entities for their abbreviations,
        # the entities on the doc may not match predicted_mention_texts.
        for entity in doc.ents:
            new_span, _ = maybe_substitute_span(doc, entity, abbreviations)
            if new_span is None:
                predicted_mention_texts.append(entity)
            else:
                # We have to manually create a new span with the new start and end points, but with the old label,
                # as spans are read only views of a document.
                span_with_label = Span(doc, start=new_span.start, end=new_span.end, label=entity.label_)
                predicted_mention_texts.append(span_with_label)

        examples_with_labels.append((doc, example, mention_texts, predicted_mention_texts, gold_umls_ids))

    print(f"Substituted {100 * substituted/total} percent of entities")
    return examples_with_labels, missing_entity_ids


def get_predicted_mention_candidates_and_types(span,
                                               ner_entities,
                                               filtered_batch_candidate_neighbor_ids,
                                               predicted_mention_types,
                                               use_soft_matching):
    """
    This function returns three lists, candidates, mention types, and mention spans. These lists will be the same length and have
    length equal to the number of predicted entities that overlap with the input gold entity. When not using soft mentions,
    this length will be equal to one, as only one predicted entity can exactly match a gold entity.
    """
    candidates = []
    mention_types = []
    mention_spans = []

    if span is not None:
        for j, predicted_entity in enumerate(ner_entities):
            if not use_soft_matching and span == predicted_entity:
                candidates.append(filtered_batch_candidate_neighbor_ids[j])
                mention_types.append(predicted_mention_types[j])
                mention_spans.append(predicted_entity)
                break
            elif use_soft_matching:
                # gold span starts inside the predicted span
                if (span.start_char <= predicted_entity.start_char <= span.end_char
                        # predicted span starts inside gold span.
                        or predicted_entity.start_char <= span.start_char <= predicted_entity.end_char):
                    candidates.append(filtered_batch_candidate_neighbor_ids[j])
                    mention_types.append(predicted_mention_types[j])
                    mention_spans.append(predicted_entity)

    return candidates, mention_types, mention_spans


def eval_candidate_generation_and_linking(examples: List[data_util.MedMentionExample],
                                          umls_concept_dict_by_id: Dict[str, Dict],
                                          candidate_generator: CandidateGenerator,
                                          k_list: List[int],
                                          thresholds: List[float],
                                          use_gold_mentions: bool,
                                          nlp: Language,
                                          generate_linking_classifier_training_data: bool,
                                          linker: Linker = None,
                                          use_soft_matching: bool = False,
                                          substitute_abbreviations: bool = False):
    """
    Evaluate candidate generation and linking using either gold mentions or spacy mentions.
    The evaluation is done both at the mention level and at the document level. If the evaluation
    is done with spacy mentions at the mention level, a pair is only considered correct if
    both the mention and the entity are exactly correct. This could potentially be relaxed, but this 
    matches the evaluation setup from the MedMentions paper.

    Parameters
    ----------
    examples: List[data_util.MedMentionExample]
        An list of MedMentionExamples being evaluted
    umls_concept_dict_by_id: Dict[str, Dict]
        A dictionary of UMLS concepts
    candidate_generator: CandidateGenerator
        A CandidateGenerator instance for generating linking candidates for mentions
    k_list: List[int]
        A list of values determining how many candidates are generated.
    thresholds: List[float]
        A list of threshold values determining the cutoff score for candidates
    use_gold_mentions: bool
        Evalute using gold mentions and types or predicted spacy ner mentions and types
    spacy_model: str
        Name (or path) of a spacy model to use for ner predictions
    generate_linking_classifier_training_data: bool
        If true, collect training data for the linking classifier
    linker: Linker
        A linker to evaluate. If None, skip linking evaluation
    use_soft_matching:
        If true, allow predicted mentions with any overlap with the gold mention to count as correct,
        else only count exact matches as correct
    substitute_abbreviations: bool
        Whether or not to substitute abbreviations when doing mention generation.
    """

    examples_with_text_and_ids, missing_entity_ids = get_mention_text_and_ids_by_doc(examples,
                                                                                     umls_concept_dict_by_id,
                                                                                     nlp,
                                                                                     substitute_abbreviations)

    linking_classifier_training_data = []
    for k in k_list:
        for threshold in thresholds:


            entity_correct_links_count = 0  # number of correctly linked entities
            entity_wrong_links_count = 0  # number of wrongly linked entities
            entity_no_links_count = 0  # number of entities that are not linked
            num_candidates = []
            num_filtered_candidates = []

            doc_entity_correct_links_count = 0  # number of correctly linked entities
            doc_entity_missed_count = 0  # number of gold entities missed
            doc_mention_no_links_count = 0  # number of ner mentions that did not have any linking candidates
            doc_num_candidates = []
            doc_num_filtered_candidates = []
            doc_linking_correct_count = 0
            doc_linking_golds_in_candidates = 0
            doc_linking_total_predictions = 0

            all_golds_per_doc_set = []
            all_golds = []
            all_mentions = []

            # Note: these counts correspond to the number of gold entities that were correctly identified using either
            # predicted mentions or gold mentions. It does not mean that these counts are only for using gold mentions
            gold_entities_linker_correct = defaultdict(int)
            gold_entities_linker_incorrect = defaultdict(int)

            predicted_entities_linker_correct = defaultdict(int)
            predicted_entities_linker_incorrect = defaultdict(int)

            for doc, example, gold_entities, predicted_entities, gold_umls_ids in tqdm(examples_with_text_and_ids,
                                                                                    desc="Iterating over examples",
                                                                                    total=len(examples_with_text_and_ids)):


                entities = [entity for entity in example.entities if entity.umls_id in umls_concept_dict_by_id]
                gold_umls_ids = [entity.umls_id for entity in entities]
                doc_golds = set(gold_umls_ids)
                doc_candidates = set()
                doc_linker_predictions = set()
                doc_all_entities_in_candidates = set()

                if use_gold_mentions:
                    mention_texts = gold_entities
                    mention_types = [[ent.mention_type] for ent in example.entities]
                else:
                    mention_types = [[ent.label_] for ent in predicted_entities]
                    mention_texts = [ent.text for ent in predicted_entities]

                batch_candidate_neighbor_ids = candidate_generator.generate_candidates(mention_texts, k)

                filtered_batch_candidate_neighbor_ids = []
                for candidate_neighbor_ids, mention_text, mention_type \
                    in zip(batch_candidate_neighbor_ids, mention_texts, mention_types):
                    # Keep only canonical entities for which at least one mention has a score less than the threshold.
                    filtered_ids = {k: v for k, v in candidate_neighbor_ids.items() if any([z[1] <= threshold for z in v])}
                    filtered_batch_candidate_neighbor_ids.append(filtered_ids)
                    num_candidates.append(len(candidate_neighbor_ids))
                    num_filtered_candidates.append(len(filtered_ids))
                    doc_candidates.update(filtered_ids)
                    # Note: doing linking here means that each entity is linked twice, which is inefficient. However the main
                    # loop below loops over gold entities, so to compute the document level metrics we first link for all predicted
                    # entities here. This could be refactored to remove the inefficiency
                    if len(filtered_ids) != 0:
                        sorted_candidate_ids = linker.link(filtered_ids, mention_text, mention_type)
                        doc_all_entities_in_candidates.update(filtered_ids)
                        doc_linker_predictions.add(sorted_candidate_ids[0])

                for i, gold_entity in enumerate(entities):
                    if use_gold_mentions:
                        candidates_by_mention = [filtered_batch_candidate_neighbor_ids[i]]  # for gold mentions, len(entities) == len(filtered_batch_candidate_neighbor_ids)
                        mention_types_by_mention = [umls_concept_dict_by_id[gold_entity.umls_id]['types']]  # use gold types
                        overlapping_mention_spans = doc.char_span(gold_entity.start, gold_entity.end)
                    else:
                        # for each gold entity, search for a corresponding predicted entity that has the same span
                        span_from_doc = doc.char_span(gold_entity.start, gold_entity.end)
                        if span_from_doc is None:
                            # one case is that the spacy span has an extra period attached to the end of it
                            span_from_doc = doc.char_span(gold_entity.start, gold_entity.end+1)

                        candidates_by_mention, mention_types_by_mention, overlapping_mention_spans = get_predicted_mention_candidates_and_types(span_from_doc, predicted_entities,
                                                                                                                                                filtered_batch_candidate_neighbor_ids,
                                                                                                                                                mention_types, use_soft_matching)
                        mention_text = ""  # not used 

                    # Evaluating candidate generation
                    if len(candidates_by_mention) == 0 or len(candidates_by_mention[0]) == 0:
                        entity_no_links_count += 1
                    elif any(gold_entity.umls_id in candidates for candidates in candidates_by_mention):
                        entity_correct_links_count += 1
                    else:
                        entity_wrong_links_count += 1

                    # Evaluating linking
                    if linker:
                        linking_predictions_by_mention = []
                        for candidates, mention_type in zip(candidates_by_mention, mention_types_by_mention):
                            sorted_candidate_ids = linker.link(candidates, mention_text, mention_type)
                            linking_predictions_by_mention.append(sorted_candidate_ids)

                        for linker_k in [1, 3, 5, 10]:
                            if any(gold_entity.umls_id in linking_predictions[:linker_k] for linking_predictions in linking_predictions_by_mention):
                                gold_entities_linker_correct[linker_k] += 1
                            else:
                                gold_entities_linker_incorrect[linker_k] += 1

                            for mention_index, linking_predictions in enumerate(linking_predictions_by_mention):
                                if gold_entity.umls_id in linking_predictions[:linker_k]:
                                    predicted_entities_linker_correct[linker_k] += 1
                                else:
                                    predicted_entities_linker_incorrect[linker_k] += 1

                    # Generate training data for the linking classifier
                    if generate_linking_classifier_training_data:
                        for candidates, mention_types_for_mention in zip(candidates_by_mention, mention_types_by_mention):
                            for candidate_id, candidate in candidates.items():
                                classifier_example = linker.classifier_example(candidate_id, candidate, mention_text, mention_types_for_mention)
                                classifier_example['label'] = int(gold_entity.umls_id == candidate_id)
                                linking_classifier_training_data.append(classifier_example)

                # the number of correct entities for a given document is the number of gold entities contained in the candidates
                # produced for that document
                doc_entity_correct_links_count += len(doc_candidates.intersection(doc_golds))
                # the number of incorrect entities for a given document is the number of gold entities not contained in the candidates
                # produced for that document
                doc_entity_missed_count += len(doc_golds - doc_candidates)

                doc_linking_correct_count += len(doc_linker_predictions.intersection(doc_golds))
                doc_linking_golds_in_candidates += len(doc_golds.intersection(doc_all_entities_in_candidates))
                doc_linking_total_predictions += len(doc_linker_predictions)

                all_golds_per_doc_set += list(doc_golds)
                all_golds += gold_umls_ids
                all_mentions += mention_texts

            print(f'K: {k}, Filtered threshold : {threshold}')
            print('Gold concept in candidates: {0:.2f}%'.format(100 * entity_correct_links_count / len(all_golds)))
            print('Gold concept not in candidates: {0:.2f}%'.format(100 * entity_wrong_links_count / len(all_golds)))
            print('Doc level gold concept in candidates: {0:.2f}%'.format(100 * doc_entity_correct_links_count / len(all_golds_per_doc_set)))
            print('Doc level gold concepts missed: {0:.2f}%'.format(100 * doc_entity_missed_count / len(all_golds_per_doc_set)))
            print('Candidate generation failed: {0:.2f}%'.format(100 * entity_no_links_count / len(all_golds)))
            if linker:
                print('Mention linking precision {0:.2f}%'.format(100 * predicted_entities_linker_correct[1] / (predicted_entities_linker_correct[1] + predicted_entities_linker_incorrect[1])))
                print('Doc linking precision {0:.2f}%'.format(100 * doc_linking_correct_count / doc_linking_total_predictions))
                print('Normalized doc linking precision {0:.2f}%'.format(100 * doc_linking_correct_count / doc_linking_golds_in_candidates))
                print('Doc linking recall {0:.2f}%'.format(100 * doc_linking_correct_count / len(all_golds_per_doc_set)))
            for linker_k in [1, 3, 5, 10]:
                correct = gold_entities_linker_correct[linker_k]
                total = len(all_golds)
                print('Linking mention-level recall@{0}: {1:.2f}%'.format(linker_k, 100 * correct / total))
                print('Normalized linking mention-level recall@{0}: {1:.2f}%'.format(linker_k, 100 * correct / entity_correct_links_count))
            print('Mean, std, min, max candidate ids: {0:.2f}, {1:.2f}, {2}, {3}'.format(np.mean(num_candidates), np.std(num_candidates), np.min(num_candidates), np.max(num_candidates)))
            print('Mean, std, min, max filtered candidate ids: {0:.2f}, {1:.2f}, {2}, {3}'.format(np.mean(num_filtered_candidates), np.std(num_filtered_candidates), np.min(num_filtered_candidates), np.max(num_filtered_candidates)))

    return linking_classifier_training_data

def main(umls_path: str,
         model_path: str,
         ks: str,
         thresholds,
         use_gold_mentions: bool = False,
         train: bool = False,
         spacy_model: str = "",
         generate_linker_data: bool = False,
         use_soft_matching: bool = False,
         substitute_abbreviations: bool = False):

    umls_concept_list = load_umls_kb(umls_path)
    # relevant_concepts = set([line.split('|')[0] for line in open('data/umls/umls/MRCONSO_DUT.RRF','r')])
    # umls_concept_list = [c for c in umls_concept_list if c['concept_id'] in relevant_concepts]
    umls_concept_dict_by_id = {c['concept_id']: c for c in umls_concept_list}
    print(f'Number of umls concepts: {len(umls_concept_list)}')

    # We need to keep around a map from text to possible canonical ids that they map to.
    text_to_concept_id: Dict[str, Set[str]] = defaultdict(set)

    for concept in umls_concept_list:
        for alias in set(concept["aliases"]).union({concept["canonical_name"]}):
            text_to_concept_id[alias].add(concept["concept_id"])

    if train:
        create_tfidf_ann_index(model_path, text_to_concept_id)
    ann_concept_aliases_list, tfidf_vectorizer, ann_index = load_tfidf_ann_index(model_path)
    candidate_generator = CandidateGenerator(ann_index, tfidf_vectorizer, ann_concept_aliases_list, text_to_concept_id, False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--umls_path',
            default="data/umls/umls_2017_aa_cat0129.json",
            help='Path to the json UMLS release.'
    )
    parser.add_argument(
            '--model_path',
            default="models/umls",
            help='Path to a directory with tfidf vectorizer and nmslib ann index.'
    )
    parser.add_argument(
            '--ks',
            help='Comma separated list of number of candidates.',
    )
    parser.add_argument(
            '--thresholds',
            default=None,
            help='Comma separated list of threshold values.',
    )
    parser.add_argument(
            '--train',
            action="store_true",
            help='Fit the tfidf vectorizer and create the ANN index.',
    )
    parser.add_argument(
            '--use_gold_mentions',
            action="store_true",
            help="Use gold mentions for evaluation rather than a model's predicted mentions"
    )
    parser.add_argument(
            '--spacy_model',
            default="",
            help="The name of the spacy model to use for evaluation (when not using gold mentions)"
    )
    parser.add_argument(
            '--generate_linker_data',
            action="store_true",
            help="Collect and save training data for the classifier."
    )

    parser.add_argument(
            '--abbreviations',
            action="store_true",
            help="Detect abbreviations when doing mention detection."
    )
    parser.add_argument(
             '--use_soft_matching',
             action="store_true",
             help="When using predicted mentions, use soft matching to allow mentions with any overlap with the gold mention to count as correct"
     )

    args = parser.parse_args()
    main(args.umls_path,
         args.model_path,
         args.ks,
         args.thresholds,
         args.use_gold_mentions,
         args.train,
         args.spacy_model,
         args.generate_linker_data,
         args.use_soft_matching,
         args.abbreviations)
