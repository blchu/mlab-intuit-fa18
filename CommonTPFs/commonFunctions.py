# Takes in a body of text returns an array of words
# Punctation like periods and semicolons are separate words
# Contractions and hyphenated words are left as is
def textToWords(l):
    #Seperate into words by splitting at spaces or line breaks
    pre = l.strip().replace('\n',' \n ').split(' ')
    #initialize list which will contain processed words
    proc = []
    for i in range(len(pre)):
        word = pre[i]
        #We don't want empty strings
        if(not word): continue
        #If there is no undesired punctuation leave as is
        if(countPunctuation(word)==0):
            proc.append(word)
            continue

        #This section of the code deals with punctuation
        #First we strip off and add separately add prefix punctuation
        while(word and isPunctuation(word[0])):
            proc.append(word[0])
            word = word[1:]

        if(not word): continue

        if(countPunctuation(word)==0):
            proc.append(word)
            continue

        #Next we check for suffix punctuation by counting
        #how punctuation we have at the end of the word
        endPunctuation = -1

        while(word[:endPunctuation] and isPunctuation(word[endPunctuation])):
            endPunctuation-=1

        endPunctuation+=1

        #We finally add the trimmed word and then the suffix punctuation
        if(word[:endPunctuation]):
            proc.append(word[:endPunctuation])
        while(endPunctuation<0):
            proc.append(word[endPunctuation])
            endPunctuation+=1
    return proc

#Counts apostrophes and hyphens as punctuation as they naturally appear in complex words
def isPunctuation(c):
    return not ((c>='a' and c<='z') or (c>='A' and c<='Z') or (c>='0' and c<='9') or (c=="'" or c=="-"))

def countPunctuation(s):
    return sum([isPunctuation(c) for c in s])

sentenceEnd = ['.','?','!','\n']

#Iterate through given texts and return the sentences from the text.
#Each element of the returned list contains all the sentences of the ith text
#Each sentence is a list of words starting with a non punctuation
def sentencesFromTexts(texts):
    textSentences = []
    for t in texts:
        sentences = []
        sentence = []
        inSentence = False
        for w in t:
            if(w in sentenceEnd):
                if(inSentence):
                    sentences.append(sentence+[w])
                    sentence = []
                    inSentence = False
            elif(w=='M'):
                sentences.append(sentence[:-1])
            else:
                if(not isPunctuation(w)): inSentence = True
                if(inSentence): sentence.append(w)
        textSentences.append(sentences)
    return textSentences

#Return a function that returns (precision,recall) considering
#the overlap of n-grams present in the two texts.
def rougeNScorer(n):
    #Calculate precision recall
    #t1 is reference text and t2 is text you are comparing to reference.
    def precisionRecall(reference,inference):
        #Initialize and populate set with all distinct n-grams of t1
        ref_grams = set()
        for i in range(len(reference)-(n-1)):
            ref_grams.add(tuple(reference[i:i+n]))
        #Initialize and populate set with all distinct n-grams of t2
        inf_grams = set()
        for i in range(len(inference)-(n-1)):
            inf_grams.add(tuple(inference[i:i+n]))
        #If either set is empty return None
        if(not (ref_grams and inf_grams)): return None
        #Find the overlap between the two sets
        overlap = len(ref_grams.intersection(inf_grams))
        #Calculate and return precision and recall
        precision = overlap/len(inf_grams)
        recall = overlap/len(ref_grams)
        return (precision,recall)
    #return function created
    return precisionRecall


