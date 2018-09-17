Machine Comprehension with Syntax, Frames, and Semantics: http://www.aclweb.org/anthology/P15-2115 <br> <br>

The objective of this paper is to assess machine comprehension with a multiple choice test where the correct answer is embedded in the input text. In addition to baseline features, the paper discusses the use of syntax, frames, coreference, and word embeddings in a max-margin learning framework. The model is a linear model which, given an input vector f(P, w, q, a) where w is a latent variable and a weight vector theta, predicts the most likely answer a* to the question q by minimizing the L2-regularized max-margin loss function. <br> <br>

The necessary feature include: <br>
Sliding window similarity to compare the word overlap between the BOW representation of the question and the current window. <br>
The minimum distance between two word occurrences in the passage that are also contained in the question-answer pair. <br>
Frame semantic features: Target, Frame Label, Argument </br>
Syntactic features: Use POS tagging and syntactic tree structure to convert question to statement, then measure similarity to sentence in the window. <br>
Word embeddings: Represent each word in a vector where similarity between vectors captures semantic similarity between words. <br> <br>

Catching the Drift: Probabilistic Content Models, with Applications to Generation and Summarization: https://arxiv.org/pdf/cs/0405039.pdf <br> <br>

The objective of this paper is to model the content structure of texts with a specific domain. The paper uses domain-specific content models (HMMs) to represent topics and changes in topics. Then the method considers Information Ordering and Extractive Summarization. <br> <br>

The content models contain states that correspond to specific topics. Each state uses alanguage model to generate sentences that are relevant to that topic. State transition probabilities encode the probability of changing from one topic to another. Forward Algorithm: compute generalization probability of a specific document. Viterbi algorithm: find most likely state sequence to have generated a document. <br> <br>

Information Ordering: Use the ordering of topics that the content model assigns the highest probability to. <br>
Extractive Summarization: Output summary is made up of the most likely sentence output by each of the states with the highest probability of having generated the sentence in the input. <br> <br>

Extractive Summarization using Continuous Vector Space Models: http://www.aclweb.org/anthology/W14-1504 <br> <br>

Similarity Measure: Represent each sentence using tf-idf and measure the cosine angle between vectors. <br> <br>

Continuous Vector Space Models: Score a set of consecutive words, distort one word, score the distorted set, then train the network to give the correct set a higher score.
