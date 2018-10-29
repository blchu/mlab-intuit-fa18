import os
import xml.etree.ElementTree as ET
from gensim.models import Word2Vec
import pickle
import sys
sys.path.insert(0,'../CommonTPFs')
from commonFunctions import *
from nltk.tokenize import PunktSentenceTokenizer

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
        return

#Load bounds and assign them appropriately
bounds = pickle.load(open("processedData/bounds.pkl",'rb'))
min_ft_sentence_count,max_ft_sentence_count = bounds[0]
min_a_word_count,max_a_word_count = bounds[1]
min_ft_avg_sentence_length,max_ft_avg_sentence_length = bounds[2]
min_ratio,max_ratio = bounds[3]
#Return if passed in texts satisfies bounds
def fitsBounds(full_text_sentences,abstract_tokens,ratio):
    #Return if v is between minimum and maximum
    inBounds = lambda v,minimum,maximum: (v>=minimum and v<=maximum) 
    #Ratio test
    if(not inBounds(ratio,min_ratio,max_ratio)): return False
    #Full text sentence count test
    num_ft_sentences = len(full_text_sentences)
    if(not inBounds(num_ft_sentences,min_ft_sentence_count,max_ft_sentence_count)): return False
    #Abstract sentence count test
    num_a_words = len(abstract_tokens)
    if(not inBounds(num_a_words,min_a_word_count,max_a_word_count)): return False
    #Average sentence length test
    avg = lambda l:sum(l)/len(l)
    avg_sentence_length = avg([len(s) for s in fullTextSentences])
    if(not inBounds(avg_sentence_length,min_ft_avg_sentence_length,max_ft_avg_sentence_length)): return False
    #If no tests failed return True
    return True


#get the parameters of the model
with open('punkt_params.pkl', 'rb') as f:
    params = pickle.load(f)

tokenizer = PunktSentenceTokenizer(params)

tokenizer._params.abbrev_types.add('dr. ')



#Initialize list whose elements will be bodies of text
texts = []
abstractSentences = []
fullTextSentences = []

#Must have folder titled 2007 to process
dataFolder = "2007/"

# Goes through all the 2007 Data and extracts summaries and texts
for month in os.listdir(dataFolder):
    #Check to make sure we are looking only at folders we are interested in
    if(not isNumerical(month)): continue
    print("Month",month)
    month+='/'
    #Create file path to all articles from a certain month
    filePath = dataFolder + month

    for day in os.listdir(filePath):
        #Check to make sure we are looking only at folders we are interested in
        if(not isNumerical(day)): continue
        print("Day",day)

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
            print(tokenizer.tokenize("Dr. Bernstien...you suck"))

            if (abstract and fullText):

                # If the abstracts and articles are non empty paths then we grab the text in them
                abstract_string = text(abstract)
                fullText_string = text(fullText)

                if (abstract_string and True or fullText_string):

                    # Get a list of scentences
                    abstract_sentence = tokenizer.tokenize(abstract_string)
                    fullText_sentence = tokenizer.tokenize(fullText_string)

                    # Get a list of words
                    abstract_tokens = textToWords(abstract_string)
                    fullText_tokens = textToWords(fullText_string)

                    # Get the ratio of the size of abstracts vs. articles
                    ratio = len(abstract_tokens) / len(fullText_tokens)

                    #Checks if text meets criteria to keep the data point
                    if(fitsBounds(fullText_sentence,abstract_tokens,ratio)):

                        fullTextSentences.append([textToWords(sentence) for sentence in fullText_sentence])
                        abstractSentences.append([textToWords(sentence) for sentence in abstract_sentence])
                        texts.append(abstract_tokens)
                        texts.append(fullText_tokens)
    #Break to only process one month.
    break

#Since every even append is an abstract and every odd append is a fullText we may extract both from texts
abstractTexts = texts[::2]
fullTextTexts = texts[1::2]


#Train Word Vectors
wordVectorSize = 50
print("Creating Word Vectors...")
wordVectors = Word2Vec(sentences=texts,size=wordVectorSize).wv
defaultWordVector = [0]*wordVectorSize

def getWordVector(w):
    if(w in wordVectors): return wordVectors[w]
    return defaultWordVector

#Generate Tokenized sentences, replace words with word vectors and display progress
n = len(abstractSentences)
tokenizedAbstractSentences = []
count = 0
for abstract in abstractSentences:
    tokenizedAbstractSentences.append([[getWordVector(w) for w in sentence]
                                        for sentence in abstract])
    count+=1
    print(f"{count}/{n} abstracts analyzed",end='\r')
print("")

tokenizedFullTextSentences = []
count = 0
for fullText in fullTextSentences:
    tokenizedFullTextSentences.append([[getWordVector(w) for w in sentence]
                                        for sentence in fullText])
    count+=1
    print(f"{count}/{n} full texts analyzed",end='\r')
print("")

#Create Directory to store processedData if it doesn't already exist
if not os.path.exists('processedData'):
    os.makedirs('processedData')

#Save Word Vectors and text to files.
print("Saving information...")
print(1)
pickle.dump(abstractTexts,open('processedData/Abstracts2007.pkl','wb'))
print(2)
pickle.dump(fullTextTexts,open('processedData/FullTexts2007.pkl','wb'))
print(3)
pickle.dump(wordVectors,open('processedData/trainedVectors2007.pkl','wb'))
print(4)
pickle.dump(abstractSentences,open('processedData/AbstractSentences2007.pkl','wb'))
print(5)
pickle.dump(fullTextSentences,open('processedData/FullTextSentences2007.pkl','wb'))
pickle.dump(tokenizedAbstractSentences,open('processedData/TokenizedAbstractSentences2007.pkl','wb'))
print(6)
pickle.dump(tokenizedFullTextSentences,open('processedData/TokenizedFullTextSentences2007.pkl','wb'))