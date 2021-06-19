"""

Convert a umls release to a json file of concepts.

"""
import json
import argparse
from preturnie.utils import umls_utils

def main(meta_path, output_path):

    concept_details = {}  # dictionary of concept_id -> {
                          #                 'concept_id': str,
                          #                 'canonical_name': str
                          #                 'aliases': List[str]
                          #                 'types': List[str]
                          #                 'definition': str
                          # }
    try:
        vocab = set([line.lstrip().rstrip().lower() for line in open('data/input/freq_vocab.txt', 'r')])
    except:
        vocab = set()

    print('Reading concepts ... ')
    umls_utils.read_umls_concepts(meta_path, concept_details, vocab=vocab)

    without_canonical_name_count = 0
    without_aliases_count = 0
    with_one_alias_count = 0
    with_more_than_one_alias_count = 0
    without_type_count = 0
    with_one_type_count = 0
    with_more_than_one_type_count = 0
    without_definition_count = 0
    with_definition_pref_source_count = 0
    with_definition_other_sources_count = 0
    for concept in concept_details.values():
        without_canonical_name_count += 1 if 'canonical_name' not in concept else 0
        without_aliases_count += 1 if len(concept['aliases']) == 0 else 0
        with_one_alias_count += 1 if len(concept['aliases']) == 1 else 0
        with_more_than_one_alias_count += 1 if len(concept['aliases']) > 1 else 0
        without_type_count += 1 if len(concept['types']) == 0 else 0
        with_one_type_count += 1 if len(concept['types']) == 1 else 0
        with_more_than_one_type_count += 1 if len(concept['types']) >= 1 else 0
        without_definition_count += 1 if 'definition' not in concept else 0
        with_definition_pref_source_count += 1 if concept.get('is_from_preferred_source') == 'Y' else 0
        with_definition_other_sources_count += 1 if concept.get('is_from_preferred_source') == 'N' else 0

    print(f'Number of concepts: {len(concept_details)}')
    print(f'Number of concepts without canonical name (one of the aliases will be used instead): '
          f'{without_canonical_name_count}')
    print(f'Number of concepts with no aliases: {without_aliases_count}')
    print(f'Number of concepts with 1 alias: {with_one_alias_count}')
    print(f'Number of concepts with > 1 alias: {with_more_than_one_alias_count}')
    print(f'Number of concepts with no type: {without_type_count}')
    print(f'Number of concepts with 1 type: {with_one_type_count}')
    print(f'Number of concepts with > 1 type: {with_more_than_one_type_count}')
    print(f'Number of concepts with no definition: {without_definition_count}')
    print(f'Number of concepts with definition from preferred sources: {with_definition_pref_source_count}')
    print(f'Number of concepts with definition from other sources: {with_definition_other_sources_count}')

    print('Deleting unused fields and choosing a canonical name from aliases ... ')
    for concept in concept_details.values():
        # if a concept doesn't have a canonical name, use the first alias instead
        if 'canonical_name' not in concept:
            aliases = concept['aliases']
            concept['canonical_name'] = aliases[0]
            del aliases[0]

        # deleting `is_from_preferred_source`
        if 'is_from_preferred_source' in concept:
            del concept['is_from_preferred_source']

    print('Exporting to the a json file {} ...'.format(output_path))
    with open(output_path, 'w') as fout:
        json.dump(list(concept_details.values()), fout, indent=4)

    print('DONE.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--meta_path',
        help="Path to the META directory of an UMLS release."
    )
    parser.add_argument(
        '--output_path',
        help="Path to the output json file"
    )
    args = parser.parse_args()
    main(args.meta_path, args.output_path)
