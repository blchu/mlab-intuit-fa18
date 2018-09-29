from CNNProcessor import model

#input number of files you want to process 1000 is default
print("Processing Text Files")
model = model(1000)
print("Getting Word Counts")
wordCounts = model.getWordCounts()
wc = wordCounts["the"]
print(f"'the' appears {wc} times")
print("Generating Word Vectors")
mapping = model.getWordVectors()
print("the vector representation of 'the'")
print(mapping["the"])
print("Words similar to Tuesday")
print(model.similar_words("Tuesday"))

'''
Check out https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors
for more cool things you can try with the word vectos.
model.getWordVectors() returns the keyedvectors object described
NOTE: default parameters are used for Word2Vec encoding.
'''