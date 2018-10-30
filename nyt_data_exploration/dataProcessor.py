import os
import xml.etree.ElementTree as ET
from gensim.models import Word2Vec
import pickle
import sys
sys.path.insert(0,'../CommonTPFs')
from commonFunctions import *

# Returns a set of all tags within a tree
# Adds current tag to set then recursively searches for more tags
def dive(root):
    n = set()
    for child in root:
        n.add(child.tag)
        x = dive(child)
        n = n.union(x)
    return n


# Tries to find the tag t anywhere with in the root tree provided
# Returns first node whose tag is the desired one or False otherwise
def findTag(root, t):
    if (root.tag == t):
        return root
    for child in root:
        ft = findTag(child, t)
        if (ft): return ft
    return False


# Tries to find the class c anywhere in the root tree provided
# Returns first node whose tag is the desired one or False otherwise
def findClass(root, c):
    if ('class' in root.attrib):
        if (root.attrib['class'] == c): return root
    for child in root:
        fc = findClass(child, c)
        if (fc): return fc
    return False

# Finds all the text associated with a given root and outputs a giant string
# Returns text of root along with text of all descendents
def text(root):
    txt = ""
    if (root.text):
        txt = root.text.strip()
        if (txt): return txt
        for child in root: txt += "\n"+text(child)
    return txt

#Returns if a string is numerical
def isNumerical(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

#Initialize list whose elements will be bodies of text
texts = []

#Must have folder titled 2007 to process
dataFolder = "2007/"

# Goes through all the 2007 Data and extracts summaries and texts
for month in os.listdir(dataFolder):
    #Check to make sure we are looking only at folders we are interested in
    if(not isNumerical(month)): continue

    month+='/'
    #Create file path to all articles from a certain month
    filePath = dataFolder + month

    for day in os.listdir(filePath):
        #Check to make sure we are looking only at folders we are interested in
        if(not isNumerical(day)): continue

        day+='/'
        #Create filepath to all articles from a certain day
        filePath = dataFolder + month + day
        for filename in os.listdir(filePath):

            # Generates a tree form the XML file of a specific article and gets root
            doc = ET.parse(filePath + filename)
            root = doc.getroot()

            # Gets the path for the abstract and articles
            abstract = findTag(root, 'abstract')
            fullText = findClass(root, 'full_text')

            if (abstract and fullText):

                # If the abstracts and articles are non empty paths then we grab the text in them
                abstract_string = text(abstract)
                fullText_string = text(fullText)

                if (abstract_string and True or fullText_string):

                    # Tokenize the text and filter out punctuation
                    abstract_tokens = textToWords(abstract_string)
                    fullText_tokens = textToWords(fullText_string)

                    # Get the ratio of the size of abstracts vs. articles
                    ratio = len(abstract_tokens) / len(fullText_tokens)

                    #Criteria to keep the data point
                    if((ratio <= 0.5) and (len(abstract_tokens) >= 5)):
                        texts.append(abstract_tokens)
                        texts.append(fullText_tokens)


#Since every even append is an abstract and every odd append is a fullText we may extract both from texts
abstractTexts = texts[::2]
fullTextTexts = texts[1::2]

#Applies previous function to all text and stores them in lists
abstractSentences = sentencesFromTexts(abstractTexts)
fullTextSentences = sentencesFromTexts(fullTextTexts)

#Train Word Vectors
wordVectorSize = 50
wordVectors = Word2Vec(sentences=texts,size=wordVectorSize).wv
defaultWordVector = [0]*wordVectorSize

def getWordVector(w):
    if(w in wordVectors): return wordVectors[w]
    return defaultWordVector

#Generate Tokenized sentences, replace words with word vectors
tokenizedAbstractSentences = [[[getWordVector(w) for w in sentence]
                                for sentence in abstract] 
                                for abstract in abstractSentences]

tokenizedFullTextentences = [[[getWordVector(w) for w in sentence]
                                for sentence in fullText] 
                                for fullText in fullTextSentences]

#Create Directory to store processedData if it doesn't already exist
if not os.path.exists('processedData'):
    os.makedirs('processedData')

#Save Word Vectors and text to files.
pickle.dump(abstractTexts,open('processedData/Abstracts2007.pkl','wb'))
pickle.dump(fullTextTexts,open('processedData/FullTexts2007.pkl','wb'))
pickle.dump(wordVectors,open('processedData/trainedVectors2007.pkl','wb'))
pickle.dump(abstractSentences,open('processedData/AbstractSentences2007.pkl','wb'))
pickle.dump(fullTextSentences,open('processedData/FullTextSentences2007.pkl','wb'))
pickle.dump(tokenizedAbstractSentences,open('processedData/AbstractSentences2007.pkl','wb'))
pickle.dump(tokenizedFullTextSentences,open('processedData/FullTextSentences2007.pkl','wb'))
