import json
import pickle
import sys
from gensim.models import Word2Vec
import os

try:
    data=sys.argv[1]
except:
    raise Exception('Please pass directory containing data')

print("Loading data...")
full_text_words = json.load(open(data+'/fulltexts.json','r'))
full_text_sentences = json.load(open(data+'/sentence_tokens.json','r'))

#Train Word Vectors
WORD_VECTOR_SIZE = 50
print("Creating Word Vectors...")
word_vectors = Word2Vec(sentences=sum(full_text_words.values(),[]),size=WORD_VECTOR_SIZE).wv

sentence_counts = [len(full_text_sentences[txt]) for txt in full_text_sentences]
#Cap is the number of sentences in the longest document.
#We use the maximum length of 80% of documents
CAP_PERCENTILE = 0.8
cap = sorted(sentence_counts)[int(len(full_text_sentences)*CAP_PERCENTILE)]

#Create Directory to store the model specific files if it doesn't already exist
if not os.path.exists('model_specific_files'):
    os.makedirs('model_specific_files')

#Save the files
print("Saving information...")

pickle.dump(word_vectors,open('model_specific_files/wv.pkl','wb'))
json.dump(cap,open('model_specific_files/cap.json','w'))
