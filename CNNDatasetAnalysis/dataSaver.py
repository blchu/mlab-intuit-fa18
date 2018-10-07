from CNNProcessor import model
import os
import pickle

#input number of files you want to process 1000 is default
#negative number means all files
print("Processing Text Files")
model = model(1000)

#Create Directory to store processedData if it doesn't already exist
if not os.path.exists('processedData'):
    os.makedirs('processedData')

print("Saving files")
#Save Word Vectors and text to files.
pickle.dump(model.summaries,open('processedData/Summaries.pkl','wb'))
pickle.dump(model.texts,open('processedData/Texts.pkl','wb'))
pickle.dump(model.getWordVectors(),open('processedData/trainedVectors.pkl','wb'))
summarySentences,textSentences = model.getSentences()
pickle.dump(summarySentences,open('processedData/summarySentences.pkl','wb'))
pickle.dump(textSentences,open('processedData/textSentences.pkl','wb'))

'''
print("Getting Word Counts")
wordCounts = model.getWordCounts()
wc = wordCounts["the"]
print(f"'the' appears {wc} times")
print("Generating Word Vectors")
mapping = model.getWordVectors()
print("the vector representation of 'the'")
print(mapping["the"])
print("Words similar to Tuesday")
print(model.similar_words("Tuesday"))'''

'''
Check out https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors
for more cool things you can try with the word vectos.
model.getWordVectors() returns the keyedvectors object described
NOTE: default parameters are used for Word2Vec encoding.
'''