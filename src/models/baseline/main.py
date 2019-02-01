#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from collections import Counter
from nltk.corpus import stopwords
import nltk
import numpy as np
from statistics import mean
import scipy.stats
import nltk
from nltk.tag import pos_tag 
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import json
import sklearn
from sklearn.linear_model import LogisticRegression

eng_stopwords = set(stopwords.words('english'))

DATA_DIR = "../../../data/"
OUTPUT_DIR = "outputs/"

# Throughout the rest of the code, it is assumed that the keys of the following dictionary
# are global variables, available to freely use. These global variables are modified only in this block. 
json_files = {'abstract_sentences': "abstracts.json", 
                'full_text_sentences': "fulltexts.json", 
                'labels': "labels.json",
                'data_splits': "data_splits.json"}   


for varname, j in json_files.items():
    qualified_name = DATA_DIR + j
    file = open(qualified_name, "rb")
    exec(varname + " = json.load(file)")
    
relevant_file_numbers = [x[0] for x in labels]

def keep_word(word):
    return word not in eng_stopwords and word != ',' and word != '.' and word != '\n'

def get_wc_map(full_text_sentences):
    word_counts_map = []
    for document in full_text_sentences.values():
    #for document in full_text_sentences: #Modified
        cnt = Counter()
        for sentence in document:
            for word in sentence:
                word = word.lower()
                if keep_word(word):
                    cnt[word] += 1
            word_counts_map.append(cnt)
    return word_counts_map

word_counts_map = get_wc_map(full_text_sentences)

def get_doc_scores(full_text_sentences, word_counts_map):
    document_sentence_scores = {} # Map of maps from sentence_id to score.
    #document_sentence_scores = [] # List of maps from sentence_id to score. #Modified
    
    for i,(document_id, document) in enumerate(full_text_sentences.items()): #Modified
        sentence_scores = {} # Map for this document.
        document_word_counts = word_counts_map[i] 
        num_doc_words = sum(document_word_counts.values())

        for sentence_id, sentence in enumerate(document):
            sentence_word_freq_sum = 0
            num_words_in_sentence = 0

            for word in sentence:
                word = word.lower()
                if keep_word(word):
                    word_freq = document_word_counts[word] / num_doc_words
                    sentence_word_freq_sum += word_freq
                    num_words_in_sentence += 1

            sentence_score = sentence_word_freq_sum / num_words_in_sentence if num_words_in_sentence != 0 else 0
            sentence_scores[sentence_id] = sentence_score 
        document_sentence_scores[document_id] = sentence_scores
        #document_sentence_scores.append(sentence_scores) #Modified
    return document_sentence_scores

document_sentence_scores = get_doc_scores(full_text_sentences, word_counts_map)

## Generating features
def sentence_position(document_num, sentence_num):
    return sentence_num / len(full_text_sentences[document_num])

def sentence_length(document_num, sentence_num, mean_sent_length=None, std_dev=5):
    sentence = full_text_sentences[document_num][sentence_num]
    return len(sentence)

def proper_noun(document_num, sentence_num):
    sentence = full_text_sentences[document_num][sentence_num]
    tagged_sent = pos_tag(sentence)
    propernouns = [word for word, pos in tagged_sent if pos == 'NNP']    
    return len(propernouns)

def sentence_freq_score(document_num, sentence_num):
    score = 1000 * document_sentence_scores[document_num][sentence_num]
    return score

features_functions = [sentence_position, sentence_length, proper_noun, sentence_freq_score]


file_numbers = [x[0] for x in labels]

## New Preprocess
train_indices = data_splits['train']
val_indices = data_splits['val']
test_indices = data_splits['test']

# Make sure these are regular lists and not numpy arrays. Things will break if they are numpy arrays.
assert type(train_indices) == list
assert type(test_indices) == list

# Create feature matrix
def create_ft_matrix(test=True, number=10):
    # Creates an X matrix with the train index file numbers' sentences appearing FIRST<
    # followed by test index file numbers' sentences.
    X = np.zeros((1, 4))
    for i in train_indices + test_indices + val_indices:
        document = full_text_sentences[i]
        for j, sentence in enumerate(document):
            X = np.vstack([X, [function(i, j) for function in features_functions]])
    X = X[1:]
    return X

def get_corr_labels(test=True, number=10):
    relevant_file_numbers = train_indices + test_indices + val_indices
    corr_labels = []
    
    for file_num, labels_list in labels.items(): #Modified
        if file_num in relevant_file_numbers:
            corr_labels.append(labels_list)
    return corr_labels

def get_num_sentences(file_numbers):
    total_num_sentences = 0
    for i in file_numbers:
        sentences = full_text_sentences[i]
        total_num_sentences += len(sentences) 
    return total_num_sentences
        

def flatten(lst):
    flattened_list = []
    for sublist in lst:
        for item in sublist:
            flattened_list.append(item)
    return flattened_list

#X = create_ft_matrix() # Took ~1 hr TODO
X = pickle.load(open("ft_matrix.p", "rb"))
corr_labels = flatten(get_corr_labels()) 
y = corr_labels

assert X.shape[0] == len(corr_labels)
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Number of sentences (i.e. rows) to use for training.
nts = get_num_sentences(train_indices)

X_train, X_test, y_train, y_test = X[0:nts], X[nts:], y[0:nts], y[nts:]

threshold = 0.15
model = LogisticRegression().fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
predictions = [1 if x[1] > threshold else 0 for x in probabilities] 
#print("Accuracy: ", accuracy_score(y_test, predictions))

assert len(probabilities) == get_num_sentences(test_indices+val_indices) # Must have a probability for each sentence.

def generate_output(probabilities):
    output = dict()
    
    used_so_far = 0
    for test_index in test_indices+val_indices:
        doc = full_text_sentences[test_index]
        ns = len(doc) # number of sentences in this document
        
        output[test_index] = [p[1] for p in probabilities[used_so_far : used_so_far + ns]]#.tolist()
        used_so_far += ns
    return output

output = generate_output(probabilities)
def save_output(output):
    json.dump(output, open(OUTPUT_DIR + "predictions.json", 'w'))

save_output(output)