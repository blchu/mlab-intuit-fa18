import numpy as np
import tensorflow as tf

#Number of sentences in document
num_sentences = 10

#Number of values used in encoding words
WORD_VECTOR_DIM = 20
#Number of values used in representing word along with surrounding context
NUM_WORD_UNITS = 20

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
avg_word_embeddings = tf.reshape(avg_word_embeddings,[1,num_sentences,NUM_WORD_UNITS*2])

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
forward_outputs = tf.reshape(forward_outputs,[num_sentences,NUM_SENTENCE_UNITS])
backward_outputs = tf.reshape(backward_outputs,[num_sentences,NUM_SENTENCE_UNITS])

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
Bias_l_nl_se = bias([num_sentences,NUM_SENTENCE_UNITS*2])

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

#Stack element tensors into one large tensor
probabilities_tensor = tf.stack(probabilities)

