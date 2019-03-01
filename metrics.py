
import re
import pandas as ps
from nltk.util import ngrams
from sortedcontainers import SortedDict
from nltk.corpus import stopwords
from functools import reduce
DEBUG = 0
BLEU_WEIGHT_NGRAM = (0.25, 0.25, 0.25, 0.25)
ROUGE_WEIGHT_NGRAM = (0.25, 0.25, 0.25, 0.25)
STOPWORDS = set(stopwords.words('english'))
"""
BLEU: precision # of overlapping words / total words in the candidate sentence
ROUGE: recall # of overlapping words / total words in target/reference summary 
"""
def preprocess(s):
    s = s.lower()
    s = re.sub(r'[^a-zA-Z0-9\s]', ' ', s)
    s = s.replace("\n"," ")
    for w in STOPWORDS:
        s = re.sub(r'\b%s\b'%w, "", s)
    return s

def freq_count(ngram_ls):
    fq_count = {}
    for g in ngram_ls:
        fq_count[g] = fq_count[g] + 1 if g in fq_count.keys() else 1
    return fq_count

"""
Assume: sum of weights are 1
"""
def geo_weighted_mean(num, weight):
    return reduce((lambda a,b: a*b), [x**y for x,y in zip(num,weight)])

"""
model_sentence: one string
target_corpus: one string
"""
def matching_grams(model_sentence, target_corpus, n):
    model = preprocess(model_sentence)
    target = preprocess(target_corpus) 
    m_token = [m for m in model.split(" ") if m != ""]
    t_token = [t for t in target.split(" ") if t != ""]
    if len(m_token) < n: ## CHECK: if the sentence is shorter than ngram
        return 0,0
    else:
        m_gram = list(ngrams(m_token, n))
        t_gram = list(ngrams(t_token, n))
        

    match_count = 0
    clip_counts = freq_count(t_gram) ## Max count for a ngram based on the its freq in the reference corpus

    for t in t_gram:
        if t in m_gram:
            match_count = match_count + 1
            #print(t)
            if match_count >= clip_counts[t]:
                continue
    BLEU_n = match_count/len(m_gram) #precision
    ROUGE_n = match_count/len(t_gram) #recall
    return BLEU_n, ROUGE_n

def sentence_score(model_sentence, target_corpus, weight_BLEU = BLEU_WEIGHT_NGRAM, weight_ROUGE = ROUGE_WEIGHT_NGRAM):
    BLEU = {}
    ROUGE = {}
    for i in range(4):
        scores = matching_grams(model_sentence, target_corpus, i+1)
        BLEU[i] = scores[0]
        if DEBUG:
            print("%d gram BLEU: %.4f" % (i+1, scores[0])) 
            print("%d gram ROUGE: %.4f" % (i+1, scores[1]))
        ROUGE[i] = scores[1]
     ## TODO: use average log or geometric mean   
    weighted_BLEU = geo_weighted_mean(BLEU.values(), weight_BLEU)
    weighted_ROUGE = geo_weighted_mean(ROUGE.values(), weight_ROUGE)
    return weighted_BLEU, weighted_ROUGE

def sentence_score_f1(model_sentence, target_corpus):
    bleu = sentence_score(model_sentence, target_corpus)[0]
    rouge = sentence_score(model_sentence, target_corpus)[1]
    return 2 * (bleu * rouge) / (bleu + rouge) if bleu > 0 and rouge > 0 else 0

def label_sentence(candidate_ls, target_corpus): #BLEU ngram weight, 
    scores = {} #key: sentence, value: f1 score
    rankings = {} #key: sentence, value: relevance ranking 1 being the most relevant
    
    for s in candidate_ls:
        scores[s] = sentence_score_f1(s, target_corpus)
    sorted_scores = sorted(scores.items(), key = lambda kv:kv[1], reverse = True)
    rankings = {x[0]:i for i,x in zip(range(len(sorted_scores)),sorted_scores)}
    return sorted_scores,rankings

"""
Testing:
"""
def within_range(x,y):
    return x-y < 0.00001 or y -x <0.00001

def get_sentence_ls(corpus):
    ls = corpus.replace("\n", " ").split(". ")
    ls = list(filter(lambda x: x != "", ls))
    return ls

def print_top_sentences_index(labeled, scores, n):
    ls = [x for x in labeled.items() if x[1] < n]
    for s in ls:
        print(s)
    return

def print_top_sentences_score(sorted, n):
    for i in range(n):
        print(sorted[i])
    return

f1 = open("test/article.txt", "r",encoding="utf-8") 
f2 = open("test/summary.txt", "r",encoding="utf-8") 
article = f1.read()
summary = f2.read()
sentence_ls = get_sentence_ls(article)
result = label_sentence(sentence_ls, summary)
#print_top_sentences_index(result[1],result[0], 5)
print_top_sentences_score(result[0],5)
