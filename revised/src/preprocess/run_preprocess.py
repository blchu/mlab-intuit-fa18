import argparse
import os
import sys
import time

sys.path.append('..')
from util.read_xml import extract_text_from_xml
from util.read_story import extract_text_from_story
from util.json_conversion import write_dict_to_json
from util.spacy_tokenizer import tokenize_document, sentence_tokenize
from labeling import get_binary_labels
from data_split import split_data
from coref import resolve

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
    parser.add_argument('--min_sentences', type=int, default=5,
                        help="Minimum number of sentences in a document")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    
    abstracts = {}
    fulltexts = {}
    sentence_tokens = {}
    labels = {}

    docs = []
    if args.source == "nyt":
        for year in os.listdir(args.root_path):
            year_path = os.path.join(args.root_path, year)
            if not os.path.isdir(year_path):
                continue
            for month in os.listdir(year_path):
                month_path = os.path.join(year_path, month)
                if not os.path.isdir(month_path):
                    continue
                for day in os.listdir(month_path):
                    day_path = os.path.join(month_path, day)
                    if not os.path.isdir(day_path):
                        continue
                    for filename in os.listdir(day_path):
                        whole_path = os.path.join(day_path, filename)
                        key = (args.source + '_' + year + '_' + month + '_' + day + '_'
                               + filename.replace('.xml', ''))
                        docs.append((whole_path, key))
    elif args.source == "cnn":
        stories_path = os.path.join(args.root_path, "stories")
        assert os.path.isdir(stories_path)
        for filename in os.listdir(stories_path):
            whole_path = os.path.join(stories_path, filename)
            key = args.source + '_' + filename.replace('.story', '')
            docs.append((whole_path, key))
    else:
        raise RuntimeError('Source is not recognized.')

    t = time.time()
    for i, (doc, key) in enumerate(docs):
        if args.source == "nyt":
            abstract_string, fulltext_string = extract_text_from_xml(doc)
        if args.source == "cnn":
            abstract_string, fulltext_string = extract_text_from_story(doc)
        if abstract_string is not None and fulltext_string is not None:
            tokenized = tokenize_document(fulltext_string)
            if args.replace:
                replace_text = resolve(fulltext_string, tokenize)
                sent_tokens = sentence_tokenize(replace_text)
            else:
                sent_tokens = sentence_tokenize(fulltext_string)
            if len(sent_tokens) > args.min_sentences:
                try:
                    labels[key] = get_binary_labels(abstract_string, sent_tokens)
                    abstracts[key] = abstract_string.strip()
                    fulltexts[key] = tokenized
                    sentence_tokens[key] = sent_tokens
                except ValueError:
                    continue
        est_time = ((time.time() - t)/(i+1))*(len(docs)-(i+1))
        print("Document {} of {} completed. Estimated time remaining: {} seconds\r"
              .format(i+1, len(docs), round(est_time,3)))

    write_dict_to_json(abstracts, os.path.join(args.output_dir, 'abstracts.json'))
    write_dict_to_json(fulltexts, os.path.join(args.output_dir, 'fulltexts.json'))
    write_dict_to_json(sentence_tokens, os.path.join(args.output_dir, 'sentence_tokens.json'))
    write_dict_to_json(labels, os.path.join(args.output_dir, 'labels.json'))
    
    key_set = set(fulltexts.keys())
    splits = split_data(key_set, num_train_subset=5000)
    write_dict_to_json(splits, os.path.join(args.output_dir, 'data_splits.json'))
