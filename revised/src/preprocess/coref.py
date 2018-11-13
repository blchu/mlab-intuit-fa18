#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coreference Resolution
"""

import en_coref_md
import re
from termcolor import colored
from copy import deepcopy

## 1. Merge into one single document string 
# Since spacy takes the whole string as resolution input, so first need to merge lsls into one document text

## hard coding for some examples of exception 
period_ls = {'Mr.': 'Mr', 'Mrs.': 'Mrs', 'Dr.': 'Dr', 'Gov.':'Gov'}

def is_punctuation(word):
    ## corner case: U.S.A
    if word == '.': 
        return True
    else:
        return False if re.match("^[a-zA-Z0-9_\-\.]*$", word) else True
    
# merge documents from a list of list of words to one document str
def merge_doc(lsls):
    doc_str = ""
    for sen in lsls:
        for word in sen:  
            ## prevent leading spaces
            if is_punctuation(word) or doc_str == "": 
                doc_str += word
            ## remove new line
            elif word != '\n':   
                doc_str += " " + word                
    return doc_str

## 2. Functions to help locate coreference clusters and their mentions:
#Since spacy gives the arbitrary position of each mention (sometimes that does not align with the word positions we have depending on how they break their sentences/words/punctuations), building a dictionary that stores both sentence start and ending positions to help search flexibly in a range.

#Note: funtions applies to a single document 

## input: list of list of words
## return: a dictionary (key: sentence positions; value: [start word pos, end word pos])
def label_positions(lsls):
    sen_pos = 0
    word_pos = 0
    label_dic = {}
    for sen in lsls:
        start = word_pos
        end = word_pos + len(sen) - 1
        label_dic[sen_pos] = [start,end]
        word_pos = end + 1
        sen_pos += 1
    return label_dic

## Input: word position
## Return: the sentence position in which the word is located
def find_sentence(dic, word_pos):
    for k,v in dic.items():
        if(word_pos >= v[0] and word_pos <= v[1]):
            return k
    return ## tolerate discrepency between model 'metion.start' index and labelled index


## Find the sublist of ls that is an exact match to the pattern
## return the first set of index of sublist
def subfinder_first(ls, pattern):
    match_index = []
    for wi in (range(len(ls))):
        for pi in (range(len(pattern))):
            if wi+pi >= len(ls) or ls[wi+pi].lower() != pattern[pi].lower():
                break
            if pi == len(pattern) - 1:
                match_index += list(range(wi,wi+len(pattern)))
                ##TODO
                if len(match_index) == 0:
                    raise IndexError("testing")
                return match_index
    return

## Resolution Rules:
#1. If the identity is in the same sentence, do not resolve;
#2. Similarly, do not resolve more than once for the same identity in the same sentence;
#3. When there are multiple choices (after passing rule 1 and 2), resolve the first reference. 
#(**Issue: may replace the instance from another cluster in the same sentence**) 
#4. Only replace references that come after the identity

#Corner cases:
#1. Replace possessive pronouns with identity + 's
#2. Replace 're with identity + 'are'
    
POSSESSIVE_PRONOUNS = ['their', 'its', 'his', 'her', 'hers', 'theirs', 'my', 'mine', 'your', 'yours']
FUZZY_SEN = 6

# doc: a single document in a str (from FullText)
# dic: labeled word pos dictionary for a single doc
# mod: spacy model for a single doc
# lsls: list of list of words (from FullTextSentences)
def resolution(mod, dic, doc, lsls):
    
    if mod._.coref_clusters is None:
        return
    for cluster in mod._.coref_clusters:
        identity = str(cluster.main)
        identity_sen = find_sentence(dic, cluster.main.start)  ## TODO: or use mention[0]
        if identity_sen is None:
            continue
        replaced_sen = []
        for ref in cluster.mentions:
            if str(ref).lower() != str(cluster.main).lower(): ## e.g. Cluster like ["He", "he"] is ignored
                sen_index = find_sentence(dic,ref.start)
                if sen_index is None:
                    continue
                
                ## Note: only replace references that come after the identity
                if sen_index > identity_sen and sen_index not in replaced_sen: 
                    possible_indices = []
                    possible_sen = []
                    for i in range(-1,FUZZY_SEN):
                        if identity_sen < sen_index+i < len(lsls) and len(lsls[sen_index+i]) > 0 : ## TODO: when to deal with empty sentences
                            possible_indices.append(sen_index+i)
                            possible_sen.append(lsls[sen_index+i])
                    
                    
                    ref_sen, id = selectReplace(possible_sen, identity, str(ref), dic)
                    if id is not None and ref_sen is not None:
                        ref_sen_index = possible_indices[id]
                        replaced_sen.append(ref_sen_index)
                    
                        ## mutate actual sentence
                        lsls[ref_sen_index] = ref_sen
    return

# sentences: a list of candidate sentences that may contain the reference
# identity: word str
# ref: word str
# dic: to update dictionary after each resolve
# return: resolved sentence as a word list, sentence index (in the list), change of word count to update dic
def selectReplace(sentences, identity, ref, dic): ## FOR DEBUGGING
    replace_str = identity  
        
    # replacing possessive pronouns
    if ref.lower() in POSSESSIVE_PRONOUNS:
        if replace_str[-2:] != "'s": ## if 's is not already in str
            replace_str = replace_str+"'s" 
    
    ##2. Locate reference to be replaced within a fuzzy range:
    ref_ls = ref.split(" ")
    replace_index = subfinder_first(sentences[0], ref_ls)
    sentence = sentences[0]
    sentence_id = 0
        
    for i in range(len(sentences)): ##TODO: while-loop
        if replace_index is not None:
            sentence_id = i
            break
        if replace_index is None:
            replace_index = subfinder_first(sentences[i], ref_ls)
            sentence = sentences[i]

    ## DEBUGGING
    if replace_index is None:
        return None, None
    
    if PRINT:
        print("###\nBefore: ") 
        mark_sentence(sentence, replace_index)

    ## Deal with messy corner cases:
    # 1. "they're" --> "[identity] are"
    if replace_index[-1]+1 < len(sentence):
        if sentence[replace_index[-1]+1] == '\'re':
            sentence[replace_index[-1]+1] = 'are'
    
    # 2. sync capitalization between ref and identity
    # At the head of the sentence
    if replace_index[0] == 0 and replace_str[0].islower():
        replace_str = replace_str[0].upper() + replace_str[1:]
    # identity looks like "The ..."
    if ref[0].islower() and replace_str.startswith("The "):
        replace_str = replace_str[0].lower() + replace_str[1:]

    ##3. Resolve
    identity_ls = replace_str.split(" ")
    rev = replace_index.copy()
    rev.reverse()
    
    for j in rev:
        del sentence[j]
    identity_ls.reverse()
    for word in identity_ls:
        sentence.insert(replace_index[0], word)
    replace_pos = range(replace_index[0], replace_index[0]+len(identity_ls))   
    
    if PRINT:
        print("After: ")   
        mark_sentence(sentence, replace_pos)
    
    return sentence, sentence_id

def mark_sentence(sen, highlight_index):
    for word, index in zip(sen,range(len(sen))):
        if(index in highlight_index):
            print(" ", colored(word, 'red'), end = "")
        elif(is_punctuation(word)):
            print(word, end= "")
        else:  
            print(" ", word, end= "")
    print("\n")
    return


# lsls: a list of list of words representation
# raw_text: a single string representation of the document
# return: a single string after resolution
def resolve(lsls, raw_text):
    lsls2 = deepcopy(lsls)
    sentence_pos  = label_positions(lsls2) 
    mod = nlp(raw_text) 
    resolution(mod, sentence_pos, raw_text, lsls2)
    resolved = merge_doc(lsls2)
    return resolved

## MACRO
nlp = en_coref_md.load()
DEBUG = 0
PRINT = 0 ## will print before and after resolution text for checking

## For testing
# CNN = resolve_all(fullTextSentences[:2])
# Note: takes ~30min for 1000 documents