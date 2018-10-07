import pickle
import matplotlib.pyplot as plt
import math
import gensim

#Load Pickled Files
abstractTexts = pickle.load(open('processedData/Abstracts2007.pkl','rb'))
fullTextTexts = pickle.load(open('processedData/FullTexts2007.pkl','rb'))
wordVectors = pickle.load(open('processedData/trainedVectors2007.pkl','rb'))
abstractSentences = pickle.load(open('processedData/AbstractSentences2007.pkl','rb'))
fullTextSentences = pickle.load(open('processedData/FullTextSentences2007.pkl','rb'))

#returns the central percentCoverage of data
def trimmedData(data,percentCoverage=0.99):
	cutoffPercent = (1-percentCoverage)/2
	cutoffNumber = int(len(data)*cutoffPercent)
	data = sorted(data)
	return data[cutoffNumber:-cutoffNumber]

#Graphs a histogram with 'bins' bins used trimmedData.
def tDensityGraph(data,percentCoverage=0.99,bins=15):
	tD = trimmedData(data,percentCoverage)
	plt.hist(tD,bins,density=True)

def tcdf(data,percentCoverage=0.99,bins=15):
	tD = trimmedData(data,percentCoverage)
	plt.hist(tD,bins,density=True,cumulative=-1)


#Creates massive list with each element indicating the length of a sentence
#Minus 1 included so as not to count end of sentence punctuation as part of sentence
sentenceLengths = []
for sentences in abstractSentences: sentenceLengths.extend([len(s)-1 for s in sentences])
for sentences in fullTextSentences: sentenceLengths.extend([len(s)-1 for s in sentences])

plt.figure(1)
plt.subplots_adjust(hspace=0.75)

plt.subplot(211)
plt.title("Sentence Length Cumulative Distribution")
tcdf(sentenceLengths)

plt.subplot(212)
plt.title("Sentence Length Density Graph")
tDensityGraph(sentenceLengths)

plt.show()


#Create a massive list with each element indicating the length of a word
wordLengths = []
for t in abstractTexts: wordLengths.extend([len(w) for w in t])
for t in fullTextTexts: wordLengths.extend([len(w) for w in t]) 

#Plot a density graph displaying the proportion of words with 1,2,...14, and 15 characters.
plt.figure(2)
plt.hist(wordLengths,bins=list(range(1,16)),density=True)
plt.title("Word Length Distribution")
plt.show()

#Create lists where each element indicates the length of a body of text.
abstractLengths = [len(t) for t in abstractTexts]
fullTextLengths = [len(t) for t in fullTextTexts]

#Graph trimmed density graphs of the length of text distributions
plt.figure(3)
plt.subplots_adjust(hspace=0.75)

plt.subplot(211)
tDensityGraph(abstractLengths)
plt.title("Summary Length Distributions")

plt.subplot(212)
tDensityGraph(fullTextLengths)
plt.title("Text Length Distributions")
plt.show()

#Graph trimmed density graph of the ratio between summary length and full text length.
plt.figure(4)

stRatio = [la/lt for la,lt in zip(abstractLengths,fullTextLengths)]
tDensityGraph(stRatio)
plt.title("Summary Length / Text Length Distribution Ratio Distribution")
plt.show()

plt.figure(5)

words = {}

for t in abstractTexts:
	for w in t:
		if(not w in words): words[w]=0
		words[w]+=1
for t in fullTextTexts:
	for w in t:
		if(not w in words): words[w]=0
		words[w]+=1

minToGraph = 100
plt.title("Histogram of word count distribution")
plt.hist(trimmedData(words.values()),bins=15,log=True)
plt.show()


print("Words most similar to Friday:")
print(wordVectors.most_similar('Friday'))
print("")

print("Words most similar to actor-man+woman:")
print(wordVectors.most_similar(positive=['actor','woman'],negative=['man']))
print("")

print("Words most similar to shirt-summer+winter:")
print(wordVectors.most_similar(positive=['shirt','winter'],negative=['summer']))
print("")
