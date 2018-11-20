from rouge import Rouge
import numpy as np

rouge = Rouge()

def rouge_sum(rouge_scores):
    rouge_metrics = ['rouge-1', 'rouge-2', 'rouge-l']
    return sum([rouge_scores[metric]['f'] for metric in rouge_metrics])


# Get binary extractive labels for each sentence in a document based off of
# an abstractive summary
def get_binary_labels(abstract, document):
    broadcasted_summaries = [abstract for _ in range(len(document))]
    scores = rouge.get_scores(document, broadcasted_summaries)
    rouge_score = [rouge_sum(score) for score in scores]
    seed_idx = np.argmax(rouge_score)
    curr_score = max(rouge_score)
    prepend_sent = ''
    sent = document[seed_idx]
    indices = set([seed_idx])
    for i in range(len(rouge_score)):
        if i < seed_idx:
            temp_sent = ' '.join([prepend_sent, document[i], sent])
            new_score = rouge_sum(rouge.get_scores(temp_sent, abstract)[0])
            if new_score > curr_score:
                curr_score = new_score
                prepend_sent = prepend_sent + ' ' + document[i]
                indices.add(i)
        if i < seed_idx:
            temp_sent = ' '.join([prepend_sent, sent, document[i]])
            new_score = rouge_sum(rouge.get_scores(temp_sent, abstract)[0])
            if new_score > curr_score:
                curr_score = new_score
                sent = sent + ' ' + document[i]
                indices.add(i)
    return [1 if i in indices else 0 for i in range(len(document))]
