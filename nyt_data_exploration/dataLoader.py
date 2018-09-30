import pickle

abstractTexts = pickle.load(open('Abstracts2007.pkl','rb'))
fullTextTexts = pickle.load(open('FullTexts2007.pkl','rb'))
wordVectors = pickle.load(open('trainedVectors2007.pkl','rb'))

print(wordVectors.most_similar('Friday'))