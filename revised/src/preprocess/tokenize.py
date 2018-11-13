import spacy
import string


nlp = spacy.load('en')

# Uses spacy to token some text at both the word and the sentence level
def tokenize_document(text):
    nlp_text = nlp(text.strip())
    sents = nlp_text.sents
    tokens = []
    while True:
        try:
            sent = next(sents)
            tokens.append([token.string.strip for token in sent
                           if str(token) not in string.whitespace])
        except StopIteration:
            break
    return tokens

# Returns the text, tokenized by sentence
def sentence_tokenize(text):
    nlp_text = nlp(text.strip())
    return [sent.string.strip() for sent in nlp_text.sents]
