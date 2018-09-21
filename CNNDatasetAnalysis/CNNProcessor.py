import os
from gensim.models import Word2Vec

class model():

	def __init__(self,nF=1000):
		self.numberFiles = nF
		self.texts = []
		self.summaries = []
		self.process()

	#Process text files from CNN dataset
	def process(self):
		fileNumber = 0
		for filename in os.listdir("./cnn_stories_tokenized/"):
			file = open("./cnn_stories_tokenized/"+filename)

			summaryCountdown = -1
			textComplete = False
			text = ""
			thisTextSummaries = []

			for line in file:
				if(line=="@highlight\n"):
					textComplete = True
					summaryCountdown = 2
				
				if(not textComplete): 
					text+=line
				else:
					if(summaryCountdown==0): thisTextSummaries.append(textToWords(line))
					summaryCountdown -=1

			self.texts.append(textToWords(text))
			self.summaries.append(thisTextSummaries)

			fileNumber +=1
			if(fileNumber==self.numberFiles): break

	def getTexts(self):
		return self.texts

	def getSummaries(self):
		return self.summaries

	def getWordCounts(self):
		if(hasattr(self,'wordCounts')): return self.wordCounts
		self.wordCounts = {}
		for text in self.texts:
			for word in text:
				if(word not in self.wordCounts): self.wordCounts[word]=0
				self.wordCounts[word]+=1
		return self.wordCounts

	def getWordVectors(self):
		if(hasattr(self,'wordVectors')): return self.wordVectors
		self.wordVectors = Word2Vec(self.texts).wv
		return self.wordVectors

	def similar_words(self,word):
		wordVectors = self.getWordVectors()
		return wordVectors.most_similar(positive=[word])


def textToWords(l):
		return l.strip().replace('\n',' ').split(' ')
