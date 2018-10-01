import os
import xml.etree.ElementTree as ET
from gensim.models import Word2Vec
import pickle

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
        for child in root: txt += text(child)
    return txt

# Takes in a body of text returns an array of words
# Punctation like periods and semicolons are separate words
# Contractions and hyphenated words are left as is
def textToWords(l):
    #Seperate into words by splitting at spaces or line breaks
    pre = l.strip().replace('\n',' ').split(' ')
    #initialize list which will contain processed words
    proc = []
    for i in range(len(pre)):
        word = pre[i]
        #We don't want empty strings
        if(len(word)==0): continue
        #If there is no undesired punctuation leave as is
        if(countPunctuation(word)==0): 
            proc.append(word)
            continue
        #If we have undesired punctuation in the middle of the string omit word
        #Otherwise add punctuation on the left or right of word separately
        if(countPunctuation(word[1:-1])==0):
            if(isPunctuation(word[0])):
                proc.append(word[0])
                word = word[1:]
            if(len(word)==0): continue
            if(not isPunctuation(word[-1])):
                proc.append(word)
            else:
                if(len(word)>1):
                    proc.append(word[:-1])
                proc.append(word[-1])
        else:
            continue
    return proc

#Counts apostrophes and hyphens as punctuation as they naturally appear in complex words
def isPunctuation(c):
    return not ((c>='a' and c<='z') or (c>='A' and c<='Z') or (c=="'" or c=="-"))
def countPunctuation(s):
    #print([isPunctuation(c) for c in s])
    return sum([isPunctuation(c) for c in s])

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

#Train Word Vectors
wordVectors = Word2Vec(texts).wv

#Create Directory to store processedData if it doesn't already exist
if not os.path.exists('processedData'):
    os.makedirs('processedData')

#Save Word Vectors and text to files.
pickle.dump(abstractTexts,open('processedData/Abstracts2007.pkl','wb'))
pickle.dump(fullTextTexts,open('processedData/FullTexts2007.pkl','wb'))
pickle.dump(wordVectors,open('processedData/trainedVectors2007.pkl','wb'))
