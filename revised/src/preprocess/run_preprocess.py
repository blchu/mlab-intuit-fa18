import argparse
import os
import sys

sys.path.append('../util')
from util.read_xml import extract_text_from_xml
from tokenize import tokenize_document, sent_tokenize
from labeling import get_binary_labels

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('root_path', required=True, type=str,
                        help="Path to data files")
    parser.add_argument('source', required=True, type=str,
                        help="Data source of summarization documents")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_arguments()
    
    abstracts = {}
    fulltexts = {}

    for year in os.listdir(args.root_path):
        year_path = os.path.join(args.root_path, year)
        for month in os.listdir(year_path):
            month_path = os.path.join(year_path, month)
            for day in os.listdir(month_path):
                day_path = os.path.join(month_path, day)
                for filename in os.listdir(day_path):
                    whole_path = os.path.join(day_path, filename)
                    abstract_string, fulltext_string = extract_text_from_xml(whole_path)
                    if abstract_string is not None and fulltext_string is not None:
                        tokenized, sent_tokens = tokenize_document(fulltext_string)
                        key = (args.source + '_' + year + '_' + month + '_' + day + '_'
                               + file.replace('.xml', ''))
                        abstracts[key] = abstract_string
                        fulltexts[key] = tokenized
                        labels[key] = get_binary_labels(abstract_string, sent_tokens)
