import pickle
import matplotlib.pyplot as plt
#import gensim

#Load Pickled Files
abstractTexts = pickle.load(open('processedData/Abstracts2007.pkl','rb'))
fullTextTexts = pickle.load(open('processedData/FullTexts2007.pkl','rb'))
wordVectors = pickle.load(open('processedData/trainedVectors2007.pkl','rb'))

#Takes the central 'percentCoverage' proportion of the data and graphs a histogram with 'bins' bins.
def trimmedDensityGraph(data,percentCoverage=0.99,bins=15):
	cutoffPercent = (1-percentCoverage)/2
	cutoffNumber = int(len(data)*cutoffPercent)
	data = sorted(data)
	trimmedData = data[cutoffNumber:-cutoffNumber]
	plt.hist(trimmedData,bins,density=True)


#Create a massive list with each element indicating the length of a word
wordLengths = []
for t in abstractTexts: wordLengths.extend([len(w) for w in t])
for t in fullTextTexts: wordLengths.extend([len(w) for w in t]) 

#Plot a density graph displaying the proportion of words with 1,2,...14, and 15 characters.
plt.figure(1)
plt.hist(wordLengths,bins=list(range(1,16)),density=True)
plt.title("Word Length Distribution")
plt.show()

#Create lists where each element indicates the length of a body of text.
abstractLengths = [len(t) for t in abstractTexts]
fullTextLengths = [len(t) for t in fullTextTexts]

#Graph trimmed density graphs of the length of text distributions
plt.figure(2)

plt.subplots_adjust(hspace=0.75)
plt.subplot(211)
trimmedDensityGraph(abstractLengths)
plt.title("Summary Length Distributions")

plt.subplot(212)
trimmedDensityGraph(fullTextLengths)
plt.title("Text Length Distributions")
plt.show()

#Graph trimmed density graph of the ratio between summary length and full text length.
plt.figure(3)

stRatio = [la/lt for la,lt in zip(abstractLengths,fullTextLengths)]
trimmedDensityGraph(stRatio)
plt.title("Summary Length / Text Length Distribution Ratio Distribution")
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
