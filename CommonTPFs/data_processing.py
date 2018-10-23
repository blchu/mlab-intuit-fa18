from rouge import Rouge
import numpy as np

import os
import pickle
import sys
import time

def get_binary_labels(sentences, summary):
    rouge = Rouge()
    broadcasted_summaries = [summary for _ in range(len(sentences))]
    scores = rouge.get_scores(sentences, broadcasted_summaries)
    rouge_l = [score['rouge-l']['f'] for score in scores]
    seed_idx = np.argmax(rouge_l)
    curr_score = max(rouge_l)
    prepend_sent = '' 
    sent = sentences[seed_idx]
    indices = set([seed_idx])
    for i in range(len(sentences)):
        if i < seed_idx:
            temp_sent = prepend_sent + ' ' + sentences[i] + ' ' + sent             
            new_score = rouge.get_scores(temp_sent, summary)[0]['rouge-l']['f']
            if new_score > curr_score:
                curr_score = new_score
                prepend_sent = prepend_sent + ' ' + sentences[i]
                indices.add(i)
        if i > seed_idx:
            temp_sent = prepend_sent + ' ' + sent + ' ' + sentences[i]
            new_score = rouge.get_scores(temp_sent, summary)[0]['rouge-l']['f']
            if new_score > curr_score:
                curr_score = new_score
                sent = sent + ' ' + sentences[i]
                indices.add(i)
    return [1 if i in indices else 0 for i in range(len(sentences))]

if __name__ == "__main__":
    with open(sys.argv[1], 'rb') as f:
        sentences = pickle.load(f)
    with open(sys.argv[2], 'rb') as f:
        abstracts = pickle.load(f)
    labels = []
    kept_docs = []
    for i in range(len(sentences)):
        if i % 100 == 0:
            print("Document %d" % i)

        document = [' '.join(sent) for sent in sentences[i]]
        summary = ' '.join(abstracts[i][0])
        try:
            labels.append(get_binary_labels(document, summary))
        except ValueError:
            continue
        kept_docs.append(sentences[i])


    with open(os.path.join(sys.argv[3], 'labels.pkl'), 'wb') as f:
        pickle.dump(labels, f)

    with open(os.path.join(sys.argv[3], 'data.pkl'), 'wb') as f:
        pickle.dump(kept_docs, f)

    print("Completed")
