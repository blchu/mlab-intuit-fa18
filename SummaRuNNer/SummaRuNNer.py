import numpy as np
import tensorflow as tf
import pickle

#Number of values used in encoding words
WORD_VECTOR_DIM = 50
#Number of values used in representing word along with surrounding context
NUM_WORD_UNITS = 20

#Number of sentences in document
num_sentences = 20

#None will be number of words in respective sentence
word_embeddings = [tf.placeholder(tf.float32,[None,WORD_VECTOR_DIM]) for _ in range(num_sentences)]

#Create Forward and Backward Gated Reccurent Units for word-level bi-RNN
word_GRU_forward = tf.nn.rnn_cell.GRUCell(NUM_WORD_UNITS)
word_GRU_backward = tf.nn.rnn_cell.GRUCell(NUM_WORD_UNITS)

avg_word_embeddings = None

#Determine average word embeddings for each sentence and populate avg_word_embeddings tensor
for i in range(num_sentences):
	#Add one in the beginning to formally dictate mini batch of size 1 (online learning)
	this_sentence_word_embeddings = tf.reshape(word_embeddings[i],[1,-1,WORD_VECTOR_DIM])

	#Get sequence of forward and backward outputs
	outputs = tf.nn.bidirectional_dynamic_rnn(cell_fw=word_GRU_forward,
											  cell_bw=word_GRU_backward,
											  inputs=this_sentence_word_embeddings,
											  dtype=tf.float32,
											  scope="word_bidirectional_RNN")[0]
	#Unpack outputs
	forward_outputs,backward_outputs = outputs

	#Both # of words x NUM_WORD_UNITS
	#Removes formality one from beginning
	#-1 is placeholder for # of words
	forward_outputs = tf.reshape(forward_outputs,[-1,NUM_WORD_UNITS])
	backward_outputs = tf.reshape(backward_outputs,[-1,NUM_WORD_UNITS])
	
	#Concatenates forward and backward outputs for each word to create # of words x (2*NUM_WORD_UNITS)
	concatenated_outputs = tf.concat([forward_outputs,backward_outputs],axis=1)
	
	#Get average word embedding by taking the element wise average between the word outputs
	#Results in a row vector of length 2*NUM_WORD_UNITS
	avg_word_embedding = tf.reshape(tf.reduce_mean(concatenated_outputs,axis=0),[1,NUM_WORD_UNITS*2])

	#Append to list of avg_word_embeddings
	if(i==0):
		avg_word_embeddings = avg_word_embedding
	else:
		avg_word_embeddings = tf.concat([avg_word_embeddings,avg_word_embedding],axis=0)

#Add one in the beginning to formally dictate mini batch of size 1 (online learning)
#Now 1 x num sentences x num word units
avg_word_embeddings = tf.reshape(avg_word_embeddings,[1,-1,NUM_WORD_UNITS*2])

NUM_SENTENCE_UNITS = 30

#Create Forward and Backward Gated Reccurent Units for sentence-level bi-RNN
sentence_GRU_forward = tf.nn.rnn_cell.GRUCell(NUM_SENTENCE_UNITS)
sentence_GRU_backward = tf.nn.rnn_cell.GRUCell(NUM_SENTENCE_UNITS)

#Get sequence of forward and backward outputs
outputs = tf.nn.bidirectional_dynamic_rnn(cell_fw=sentence_GRU_forward,
										  cell_bw=sentence_GRU_backward,
										  inputs=avg_word_embeddings,
										  dtype=tf.float32,
										  scope="sentence_bidirectional_RNN")[0]
#Unpack outputs
forward_outputs,backward_outputs = outputs

#Both num_sentences x NUM_SENTENCE_UNITS
#Removes formality one from beginning
forward_outputs = tf.reshape(forward_outputs,[-1,NUM_SENTENCE_UNITS])
backward_outputs = tf.reshape(backward_outputs,[-1,NUM_SENTENCE_UNITS])

#Concatenate forward and backward outputs to get list of sentence embeddings
sentence_embeddings = tf.concat([forward_outputs,backward_outputs],axis=1)

#Get average sentence embedding by taking the element wise average between sentence embeddings.
#Tensor vector of length 2*(NUM_SENTENCE_UNITS)
avg_sentence_embedding = tf.reshape(tf.reduce_mean(sentence_embeddings,axis=0),[2*NUM_SENTENCE_UNITS,1])

#Number of values used to embed whole document
DOC_EMBEDDING_LENGTH = 50

#Return TensorFlow weight normally initialized with standard dev 0.1
def weight(shape):
	return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

#Return TensorFlow bias with all values initialized to 0.0
def bias(shape):
	return tf.Variable(tf.constant(0.0,shape=shape))

#Weight matrix that is used in transforming avg_sentence_embedding into document_embedding
Weight_asw_de = weight([DOC_EMBEDDING_LENGTH,2*NUM_SENTENCE_UNITS])

#Bias vector that is used in transforming avg_sentence_embedding into document_embedding
Bias_asw_de = bias([DOC_EMBEDDING_LENGTH,1])

#tanh of weight x avg embedding + bias.
#Computes document_embedding.  Results size is length [DOC_EMBEDDING_LENGTH,1]
d = tf.tanh(tf.add(tf.matmul(Weight_asw_de,avg_sentence_embedding),Bias_asw_de))

#Weight matrix used in performing non linear transformation of linear sentence embeddings
Weight_l_nl_se = weight([NUM_SENTENCE_UNITS*2,NUM_SENTENCE_UNITS*2])

#Bias Vector used in performing non linear transformation of linear sentence embeddings
Bias_l_nl_se = bias([1,NUM_SENTENCE_UNITS*2])
#Same bias accros various sentences
Bias_l_nl_se = tf.tile(input=Bias_l_nl_se,multiples=[num_sentences,1])

#tanh of weight x sent embedding + bias for each sentence
#Compute nonlinear sentence embeddings.  Resulting size is [num_sentences,2*NUM_SENTENCE_UNITS]
non_linear_ses = tf.tanh(tf.add(tf.matmul(sentence_embeddings,Weight_l_nl_se),Bias_l_nl_se))

#Weights used in determining probability of sentence being in summary
#Absolute and relative position left out of calculation for now

#Content weight
Weight_content = weight([1,NUM_SENTENCE_UNITS*2])

#Salience weight
Weight_salience = weight([NUM_SENTENCE_UNITS*2,DOC_EMBEDDING_LENGTH])

#Novelty weight
Weight_novelty = weight([NUM_SENTENCE_UNITS*2,NUM_SENTENCE_UNITS*2])

#Probability bias
Bias_p = tf.Variable(0.0)

#Usage of weights explained in below implementation

#dynamic summary representation.  Will be updated to track information already captured in summary
s = tf.zeros([NUM_SENTENCE_UNITS*2,1])
#Will keep track of individual summary contributions allowing us to keep only information we need
indiv_s = None

probabilities = []

#Sequentially compute probability that sentence is in summary
for i in range(num_sentences):

	#nonlinear sentence embedding rotated to vector form of length NUM_SENTENCE_UNITS*2
	h = tf.reshape(non_linear_ses[i],[NUM_SENTENCE_UNITS*2,1])

	#Calculate content contribution of sentence.  Meant to signify how much important information is within a sentence.
	content_contribution = tf.matmul(Weight_content,h)

	#Transpose of h. Used in following matrix multiplications.
	hT = tf.transpose(h)

	#Calculate salience contribution.  Meant to signify how relevant information in sentence is to doucument as a whole.
	#Calculated using both document embedding and sentence embedding to understand their relationship.
	salience_contribution = tf.matmul(hT,tf.matmul(Weight_salience,d))

	#Calcuate novelty contribution.  Meant to signify the amount of new information in sentence not in previous sentences
	#already added to summary.  Since everything is in terms of probability previous sentences are weighted by the probability
	#they appear in the summary.  Contribution is negative because repition is not desired.
	novelty_contribution = -tf.matmul(hT,tf.matmul(Weight_novelty,tf.tanh(s)))

	#Sums contributions and bias to create overall sigmoid input
	sigmoid_input = content_contribution+salience_contribution+novelty_contribution+Bias_p

	#Computes probability sentence is in summary
	p = tf.sigmoid(sigmoid_input)

	#Adds p to list of probabilities
	probabilities.append(p)

	#Update dynamic summary representation vector
	s = tf.add(s,p*h)

	if(i==0):
		indiv_s = p*h
	else:
		indiv_s = tf.concat([indiv_s,p*h],axis=1)

#Place holder which will be [True,...,True,False,...,False] number of Trues = sentence word count
sentences_present = tf.placeholder(tf.bool,[num_sentences])
#Mask is used to remove individual summary contributions from filler sentences
indiv_s = tf.boolean_mask(indiv_s,sentences_present,axis=1)
#Summary Vector is updated to only include information from sentences that actually were in document
s = tf.reshape(tf.reduce_mean(indiv_s,axis=1),[-1,1])
#IMPORTANT: Reason for using filler sentences is that it has minimal effect on proper content and allows
#Tensorflow to create map.  At word level sentences are independent and so there is obviously no effect.
#How ever during reverse sentence level RNN there will be a percieved effect.  However due to filler sentences
#Being 0 vectors the avg_word_embeddings for that sentence will be 0.  Thus the effect of the filler sentences
#in the reverse RNN will be predictable and consistent and simply an indication of how short a text is.



#Stack element tensors into one large tensor
probabilities_tensor = tf.boolean_mask(tf.stack(probabilities),sentences_present)

#Number of hidden units in summary RNN
NUM_SUMMARY_UNITS = 25

#Number of words in summary
num_words_summary = tf.placeholder(tf.int32)

#Used in generating hidden states for summary words
abstractive_training_GRU = tf.nn.rnn_cell.GRUCell(NUM_SUMMARY_UNITS)

#initialize hidden state based on summary to internalize context
Weight_rnn_sum = weight([NUM_SUMMARY_UNITS,NUM_SENTENCE_UNITS*2])
#Transposing NUM_SUMMARY_UNITS x 1 resulting vector
initial_rnn_state = tf.reshape(tf.matmul(Weight_rnn_sum,s),[1,NUM_SUMMARY_UNITS])

#Word embeddings for each word
summary_word_embeddings = tf.placeholder(tf.float32,[None,WORD_VECTOR_DIM])

#Hidden representations of each word from summary: num_words_summary x NUM_SUMMARY_UNITS
hiddenStates,_ = tf.nn.dynamic_rnn(cell=abstractive_training_GRU,
								inputs=tf.reshape(summary_word_embeddings,[1,-1,WORD_VECTOR_DIM]),
								initial_state=initial_rnn_state,
								dtype=tf.float32,
								scope="summary_words_RNN")

hiddenStates = tf.reshape(hiddenStates,[-1,NUM_SUMMARY_UNITS])

#Dimensionality of fully connected layer
NUM_FC_UNITS = 20

#Weights and bias used in constructing fully connected layer which takes input from
#hidden values, directly from word embedding, and summary vector
Weight_fc_hidden = weight([NUM_SUMMARY_UNITS,NUM_FC_UNITS])
Weight_fc_we = weight([WORD_VECTOR_DIM,NUM_FC_UNITS])
Weight_fc_sum = weight([NUM_SENTENCE_UNITS*2,NUM_FC_UNITS])

Bias_fc = bias([NUM_FC_UNITS,1])

#Compute fully connected layer by finding contributions and applying tanh nonlinearity

hidden_component = tf.matmul(hiddenStates,Weight_fc_hidden)
word_component = tf.matmul(summary_word_embeddings,Weight_fc_we)

#Since summary component is constant across timesteps we duplicate num_words_summary of s
s = tf.reshape(s,[1,NUM_SENTENCE_UNITS*2])
s = tf.tile(input=s,multiples=[num_words_summary,1])

sum_component = tf.matmul(s,Weight_fc_sum)

#Overall dimensionality is num_words_summary x NUM_FC_UNITS
FC = tf.tanh(hidden_component+word_component+sum_component)

#Load vocabulary length to use for dimensionality of probability distribution
#Add one to account for "unknown" taking up an extra index
NUM_VOCAB = pickle.load(open('processedData/vocabLength.pkl','rb'))+1

#Weight and bias used to predict prob distribution of word in summary
Weight_softmax = weight([NUM_FC_UNITS,NUM_VOCAB])

Bias_softmax = bias([1,NUM_VOCAB])
#Duplicate bias since bias is constant across time steps
Bias_softmax = tf.tile(input=Bias_softmax,multiples=[num_words_summary,1])

#Find softmax inputs from Fully Connected layer
#Dimensionality is num_words_summary x NUM_VOCAB
softmax_inputs = tf.matmul(FC,Weight_softmax)+Bias_softmax
#Tensor of length num_summary_words representing indexes of each word.
summary_word_indexes = tf.placeholder(tf.int32,[None])
#Calculate loss using cross entropy loss at each time step and aggregating.
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=summary_word_indexes,
													    logits=softmax_inputs)
#Aggregate to arrive at numerical loss
loss = tf.reduce_mean(losses)

print("Loading data...")
#Load tokenized data and summary one-hot encodings
tokenized_abstract_sentences = pickle.load(open('processedData/TokenizedAbstractSentences2007.pkl','rb'))
tokenized_full_text_sentences = pickle.load(open('processedData/TokenizedFullTextSentences2007.pkl','rb'))
summary_one_hot_sentences = pickle.load(open('processedData/summaryEncodings.pkl','rb'))

#Aggregate between sentneces in single summaries

#Result is num abstracts x num words x word vector dim
tokenized_abstracts = [sum([s for s in abstract],[]) 
					   for abstract in tokenized_abstract_sentences]
#Result is num abstracts x num words x num vocab
summary_one_hot_encodings = [sum([s for s in abstract],[])
					 for abstract in summary_one_hot_sentences]
#Zip together various data for easy access
zipped_data = list(zip(tokenized_abstracts,tokenized_full_text_sentences,summary_one_hot_encodings))

#Go through data and return feed_dict to run graph
def get_feed_dict(abstract,full_text,summary_ohe):
	#Remove 0 length sentences from full_text
	i = 0
	while(i<len(full_text)):
		if(len(full_text[i])==0):full_text.pop(i)
		i+=1
	#Initialize padded_text to the original text
	padded_text = list(full_text)
	#Mask is used to allow network to toss out filler sentences later
	mask = [True]*min(len(full_text),num_sentences)
	#Filler sentence equivalent to sentence with one word whose word embedding is the zero vector
	#This allows for consistent and predictable behaviour on rest of netowork.
	fillerSentence = [[0]*WORD_VECTOR_DIM]
	#Pad text if necessary and complete mask with False values where fillers are present
	while(len(padded_text)<num_sentences):
		padded_text.append(fillerSentence)
		mask.append(False)
	#Assign sentences from text to placeholders
	#Note if the there are more sentences in the text than num_sentences those sentences will not be considered.
	fd = {placeholder:sentence for placeholder,sentence in zip(word_embeddings,padded_text)}
	#Provide values for other placeholders
	fd.update({summary_word_embeddings:abstract,
			   summary_word_indexes:summary_ohe,
			   num_words_summary:len(abstract),
			   sentences_present:mask})
	#Return feed dict
	return fd

#Initialize TensorFlow variables and session
sess = tf.Session()

#Return the average loss of the first num_texts texts
def averageLoss(num_texts):
	#Variable to keep track of loss
	l = 0
	#Count of texts used so far
	i = 0
	#Iterate through texts
	for abstract,full_text,summary_ohe in zipped_data:
		#Calculate loss for this text and add
		l+=sess.run(loss,feed_dict=get_feed_dict(abstract,full_text,summary_ohe))
		#Break once use number of texts have been used
		i+=1
		if(i>=num_texts): break
	return l/i

#Create training algorithm using LEARNING_RATE and set EPOCHS
LEARNING_RATE = 5e-3
print("Creating training algorithm...")
train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
sess.run(tf.global_variables_initializer())
EPOCHS = 1

#Iterate EPCOCHS number of times
for e in range(EPOCHS):
	print(f"Epoch {e+1}")
	#Amount of texts to use
	use = 3000
	#How many steps to do before printing average loss
	PRINT_EVERY = 1000
	#Count of texts used so far
	count = 0
	#Iterate through data untill count
	for abstract,full_text,summary_ohe in zipped_data:
		print(f"Train {count}",end='\r')
		train_step.run(feed_dict=get_feed_dict(abstract,full_text,summary_ohe),session=sess)
		if(count%PRINT_EVERY==0):
			print("")
			print(f"Loss: {averageLoss(use)}")
		#Break once use number of texts have been used
		count+=1
		if(count>=use): break
	print("")
	print(f"Loss: {averageLoss(use)}")


