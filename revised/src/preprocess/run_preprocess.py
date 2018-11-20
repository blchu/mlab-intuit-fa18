import argparse
import os
import sys

sys.path.append('..')
from util.read_xml import extract_text_from_xml
from util.json_conversion import write_dict_to_json
from util.spacy_tokenizer import tokenize_document, sentence_tokenize
from labeling import get_binary_labels
from data_split import split_data
# from coref import resolve

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', type=str,
                        help="Path to data files")
    parser.add_argument('source', type=str,
                        help="Data source of summarization documents")
    parser.add_argument('output_dir', type=str,
                        help="Directory of output files")
    parser.add_argument('--replace', action='store_true',
                        help="Flag for optionally replacing pronouns for label generation")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    
    abstracts = {}
    fulltexts = {}
    sentence_tokens = {}
    labels = {}

    for year in os.listdir(args.root_path):
        year_path = os.path.join(args.root_path, year)
        for month in os.listdir(year_path)[:1]:
            month_path = os.path.join(year_path, month)
            for day in os.listdir(month_path)[:1]:
                day_path = os.path.join(month_path, day)
                for filename in os.listdir(day_path):
                    whole_path = os.path.join(day_path, filename)
                    abstract_string, fulltext_string = extract_text_from_xml(whole_path)
                    if abstract_string is not None and fulltext_string is not None:
                        tokenized = tokenize_document(fulltext_string)
                        key = (args.source + '_' + year + '_' + month + '_' + day + '_'
                               + filename.replace('.xml', ''))
                        abstracts[key] = abstract_string.strip()
                        fulltexts[key] = tokenized
                        if args.replace:
                            replace_text = resolve(fulltext_string, tokenize)
                            sent_tokens = sentence_tokenize(replace_text)
                        else:
                            sent_tokens = sentence_tokenize(fulltext_string)
                        labels[key] = get_binary_labels(abstract_string, sent_tokens)
                        sentence_tokens[key] = sent_tokens
                print("Documents in %s/%s/%s processed" % (month, day, year))

    write_dict_to_json(abstracts, os.path.join(args.output_dir, 'abstracts.json'))
    write_dict_to_json(fulltexts, os.path.join(args.output_dir, 'fulltexts.json'))
    write_dict_to_json(sentence_tokens, os.path.join(args.output_dir, 'sentence_tokens.json'))
    write_dict_to_json(labels, os.path.join(args.output_dir, 'labels.json'))
    
    key_set = set(fulltexts.keys())
    splits = split_data(key_set)
    write_dict_to_json(splits, os.path.join(args.output_dir, 'data_splits.json'))
