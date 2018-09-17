	Simple and EFfective Multi-Paragraph Reading Comprehension

	-Teaching machines to answer arbitrary user-generated questions. 
	-Two types of approaches: 
	--pipelined (select useful paragraph, and use a paragraph-model to extract answer
	)
	--confidence: apply paragraph model to multiple paragraphs and then pick the best


	Piplined
	-TF-IDF heuristic. Choose the paragraph with the smallest TF-IDF cosine distance with the question. Term frequency-inverse document frequency. 
	-Document frequencies are computed using just the paragraphs, not the entire text.
	--Advantage of this is, for instance, if the word "tiger" appears many times in the corpus and the question is "smallest tiger", then the word "smallest" is given as much weight.

	-Note that annotating entire documents is difficult, so the training data is just the question and the answer, not the question, answer and the area from where the answer is extracted. Uses a negative log probability model across the different possible spans to pick the best 'paragraph'. 

	This paper is basically just a reduction to the bidaf paper...


	*Confidence
	Four different techniques experimented
	1. Shared-normalization. All paragraphs are processed independently. The objective function used shares the normalization factor between all paragraphs. This is similar to simply feeding the model multiple paragraphs from each context concatenated together, except each paragraph is processed independently until normalization. 
	2. Merge: actually concatenate the paragraph together (with a paragraph separator token).
	3. Allow for "No answer": Add some binary variables in your objective function. Use a hidden layer to determine z using the span-start scores and the span-end scores. 'z' is the weight given to the no-answer possibility.
	4. Use a sigmoid loss function instead of negative log loss. 





	Bi-Directional Attention Flow For Machine Comprehension

	Answering query about a given context paragraph. 

	6 layers
	1.Character Embedding Layer maps each word to a vector space using character-level
	CNNs. Uses a CNN.
	2. Word Embedding Layer maps each word to a vector space using a pre-trained word embedding
	model. Uses GloVe.
	3. Contextual Embedding Layer utilizes contextual cues from surrounding words to refine
	the embedding of the words. These first three layers are applied to both the query and
	context. Uses LSTM on top of the word embeddings to preserve temporal interactions. This is from both -- context paragraph and query.
	4. Attention Flow Layer couples the query and context vectors and produces a set of query aware
	feature vectors for each word in the context. This is NOT used to summarize the query and context into single feature vectors. (No loss by early summarization). Attention is computed from context to query and from query to context. $\mathbf{S}_{tj}$ is a trainable scalar function that encodes similarity between t-th context word and j-th query word. The attentions are derived from this similarity matrix. Some linear combination of softmax is applied to form the c-to-q attention and q-to-c attention, and then the final result is a query-aware representation of context-words. 

	5. Modeling Layer employs a Recurrent Neural Network to scan the context. Bi-directional LTSM, to get a matrix. The matrix contains contextual information about the word with respect to the entire context paragraph and the query.

	6. Output Layer provides an answer to the query. This can be easily swapped out based on task. The phrase is derived by predicting the start and end indices. This is done using softmax on the modelling layer and the matrix with the contexual information. 

	Training loss: sum of negative log probabilities of the true start and end indices. 
