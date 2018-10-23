## A Review on Automatic Text Summarization Approaches
http://thescipub.com/PDF/jcssp.2016.178.190.pdf  
Summary of paper:  
Notable results:  
What we can use for project:  

## A Neural Attention Model for Abstractive Sentence Summarization
https://arxiv.org/pdf/1509.00685.pdf  
Summary of paper:  
Notable results:  
What we can use for project:  

## Extractive Summarization using Deep Learning
https://arxiv.org/pdf/1708.04439.pdf  
The paper extracts features from a cleaned version of the text, and then uses a Restricted Boltzman Machine to determine which segments of text to keep for the summary. 
First they preprocesses all the data with stopword removal, lemmatization, PoS tagging, and segmentation. 
A feature vector is taken for each cleaned sentence, the featurization is as such: num thematic words, position, length, paragraph position, num named entities, tf-isf, centroid similarity. 
A RBM is used to enhance the features, which is trained fully for each document to be summarized. 
Sentences are scored by summing the enhanced features and then are ranked based off of highest score.
