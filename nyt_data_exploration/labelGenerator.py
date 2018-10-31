import pickle

word_vectors = pickle.load(open('processedData/trainedVectors2007.pkl','rb'))
#Dictionary that will ultimately map words to indexes
word_to_index = {}
#Arbitrarily assign indexes from 1 to numVocab to each word in vocab
for i,w in enumerate(word_vectors.vocab):
	word_to_index[w]=i+1

abstract_sentences = pickle.load(open('processedData/AbstractSentences2007.pkl','rb'))
#Replace words in summaries with one hot representations to be used in training SummaRuNNer
#O is used in representing infrequent words
summary_encodings = [[[word_to_index.get(w,0) for w in sentence]
                       for sentence in abstract]
                       for abstract in abstract_sentences]
pickle.dump(summary_encodings,open('processedData/summaryEncodings.pkl','wb'))
pickle.dump(len(word_vectors.vocab),open('processedData/vocabLength.pkl','wb'))

full_texts = pickle.load(open('processedData/tokenizedFullTextSentences2007.pkl','rb'))
#Create an list of all sentence counts
sentence_counts = [len(txt) for txt in full_texts]
#Cap is the number of sentences in the longest document.
cap = max(sentence_counts)
pickle.dump(cap,open('processedData/cap.pkl','wb'))