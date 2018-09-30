import os
import xml.etree.ElementTree as ET
from gensim.models import Word2Vec
import pickle

filePath = "./2007"

# returns a set of all possible tags within a tree
def dive(root):
    n = set()
    for child in root:
        n.add(child.tag)
        x = dive(child)
        n = n.union(x)
    return n


# Tries to find the tag t anywhere with in the root tree provided
def findTag(root, t):
    if (root.tag == t):
        return root
    for child in root:
        ft = findTag(child, t)
        if (ft): return ft
    return False


# Tries to find the class c anywhere in the root tree provided
def findClass(root, c):
    if ('class' in root.attrib):
        if (root.attrib['class'] == c): return root
    for child in root:
        fc = findClass(child, c)
        if (fc): return fc
    return False

# Finds all the text associated with a given root and outputs a giant string
def text(root):

    txt = ""

    if (root.text):
        txt = root.text.strip()
        if (txt): return txt
        for child in root: txt += text(child)
    return txt

# Takes in a body of text returns an array of words
# Punctation like periods and semicolons are separate words
# Contractions are left as is
def textToWords(l):
        pre = l.strip().replace('\n',' ').split(' ')
        proc = []
        for i in range(len(pre)):
            word = pre[i]
            if(len(word)==0): continue
            if(countPunctuation(word)==0): 
                proc.append(word)
                continue
            if(countPunctuation(word[1:-1])==0):
                if(isPunctuation(word[0])):
                    proc.append(word[0])
                    word = word[1:]
                if(len(word)==0): continue
                if(not isPunctuation(word[-1])):
                    proc.append(word)
                else:
                    proc.append(word[:-1])
                    if(len(word)==0): continue
                    proc.append(word[-1])
            else:
                continue
        return proc


def isPunctuation(c):
    return not ((c>='a' and c<='z') or (c>='A' and c<='Z') or (c=="'"))
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


texts = []

filePath = "./2007"

# Goes through all the 2007 Data
for month in os.listdir(filePath):
    if(not isNumerical(month)): continue
    filePath = "./2007/" + month
    for day in os.listdir(filePath):
        if(not isNumerical(day)): continue
        filePath = "./2007/" + month + "/" + day
        for filename in os.listdir(filePath):
            # Generates a tree form the XML file
            doc = ET.parse(filePath + "/" + filename)
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
                    #Current criteria to keep the data point
                    if((ratio <= 0.5) and (len(abstract_tokens) >= 5)):
                        texts.append(abstract_tokens)
                        texts.append(fullText_tokens)

abstractTexts = texts[::2]
fullTextTexts = texts[1::2]


wordVectors = Word2Vec(texts).wv



pickle.dump(abstractTexts,open('Abstracts2007.pkl','wb'))
pickle.dump(fullTextTexts,open('FullTexts2007.pkl','wb'))
pickle.dump(wordVectors,open('trainedVectors2007.pkl','wb'))
