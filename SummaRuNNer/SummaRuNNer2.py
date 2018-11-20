import numpy as np
import tensorflow as tf
import pickle
import sys
sys.path.insert(0,'../CommonTPFs')
from commonFunctions import *
import matplotlib.pyplot as plt
from rouge import Rouge

#Specifiy training method
#Adds additional sequential model to be used in training
ABSTRACTIVE = False
#Trains directly of probability dist using labels given
EXTRACTIVE = not ABSTRACTIVE
#variable specificies whether to start from scratch
#or initialize using prior trained weights
USE_LAST = True
#0->Train, 1->Use
MODE = 1

#Number of values used in encoding words
WORD_VECTOR_DIM = 50
#Number of values used in representing word along with surrounding context
NUM_WORD_UNITS = 20

#Number of sentences to be analyzed.  If document has below NUM_SENTENCES sentences
#Trivial sentences (one word 0 word vector embedding) will be used to pad up to NUM_SENTENCES
#If document has more the NUM_SENTENCES sentences on the first NUM_SENTENCES only the first
#NUM_SENTENCES sentences will be assigned probabilites for being in the summary.
NUM_SENTENCES = pickle.load(open("processedData/cap.pkl",'rb'))
print(f"NUM_SENTENCES: {NUM_SENTENCES}")
print("Creating model...")
#None will be number of words in respective sentence
word_embeddings = [tf.placeholder(tf.float32,[None,WORD_VECTOR_DIM]) for _ in range(NUM_SENTENCES)]

#Create Forward and Backward Gated Reccurent Units for word-level bi-RNN
word_GRU_forward = tf.nn.rnn_cell.GRUCell(NUM_WORD_UNITS)
word_GRU_backward = tf.nn.rnn_cell.GRUCell(NUM_WORD_UNITS)

avg_word_embeddings = None

#Determine average word embeddings for each sentence and populate avg_word_embeddings tensor
for i in range(NUM_SENTENCES):
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

#Both NUM_SENTENCES x NUM_SENTENCE_UNITS
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
Bias_l_nl_se = tf.tile(input=Bias_l_nl_se,multiples=[NUM_SENTENCES,1])

#tanh of weight x sent embedding + bias for each sentence
#Compute nonlinear sentence embeddings.  Resulting size is [NUM_SENTENCES,2*NUM_SENTENCE_UNITS]
non_linear_ses = tf.tanh(tf.add(tf.matmul(sentence_embeddings,Weight_l_nl_se),Bias_l_nl_se))

#Weights used in determining probability of sentence being in summary
#Absolute and relative position left out of calculation for now

#Content weight
Weight_content = weight([1,NUM_SENTENCE_UNITS*2])

#Salience weight
Weight_salience = weight([NUM_SENTENCE_UNITS*2,DOC_EMBEDDING_LENGTH])

#Novelty weight
Weight_novelty = weight([NUM_SENTENCE_UNITS*2,NUM_SENTENCE_UNITS*2])

#Position weight
Weight_Position = weight([1,1])

#Probability bias
Bias_p = tf.Variable(0.0)

#Usage of weights explained in below implementation

#dynamic summary representation.  Will be updated to track information already captured in summary
s = tf.zeros([NUM_SENTENCE_UNITS*2,1])
#Will keep track of individual summary contributions allowing us to keep only information we need
indiv_s = None

probabilities = []

#Sequentially compute probability that sentence is in summary
for i in range(NUM_SENTENCES):

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

	#Allows the network to consider the absolute position of the sentence in it's prediction
	#We scale a weight by log(1+i) where i is the position
	position_contribution = tf.log1p(tf.cast(i,tf.float32))*Weight_Position

	#Sums contributions and bias to create overall sigmoid input
	sigmoid_input = content_contribution+salience_contribution+novelty_contribution+position_contribution+Bias_p

	#Computes probability sentence is in summary
	p = tf.sigmoid(sigmoid_input)

	#Adds p to list of probabilities
	#p will bea 1x1 tensor this accounted for later.
	probabilities.append(p)

	#Update dynamic summary representation vector
	s = tf.add(s,p*h)

	if(i==0):
		indiv_s = p*h
	else:
		indiv_s = tf.concat([indiv_s,p*h],axis=1)

#Place holder which will be [True,...,True,False,...,False] number of Trues = sentence word count
sentences_present = tf.placeholder(tf.bool,[NUM_SENTENCES])
#Mask is used to remove individual summary contributions from filler sentences
indiv_s = tf.boolean_mask(indiv_s,sentences_present,axis=1)
#Summary Vector is updated to only include information from sentences that actually were in document
s = tf.reshape(tf.reduce_mean(indiv_s,axis=1),[-1,1])
#IMPORTANT: Reason for using filler sentences is that it has minimal effect on proper content and allows
#Tensorflow to create map.  At word level sentences are independent and so there is obviously no effect.
#How ever during reverse sentence level RNN there will be a percieved effect.  However due to filler sentences
#Being 0 vectors the avg_word_embeddings for that sentence will be 0.  Thus the effect of the filler sentences
#in the reverse RNN will be predictable and consistent and simply an indication of how short a text is.


#Stack element tensors into one large 1D tensor of length = number of sentences present
probabilities_tensor = tf.reshape(tf.boolean_mask(tf.stack(probabilities),sentences_present),[-1])

#Takes in probabilities for each sentence and corresponding text
#Returns a string containing each sentence for which the corresponding
#probability is greater than 0.5
def generateSummary(probabilities,text,threshold=0.1):
	summary = []
	#iterate through probabilities
	for i in range(len(probabilities)):
		#if probability>threshold append sentence to summary
		if(probabilities[i]>threshold): summary+=text[i]
	return summary

#General train procedure
#data paramater represents a list.  Each element is a dict.
#Each dict has keys that map to attributes of the data depending
#on which specific method
#gfd is a function that takes an element of data and returns and appropriate
#feed-dict depending on the method
def train(data,gfd):

	#Initialize the session
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
	LEARNING_RATE = 4e-3
	print("Creating training algorithm...")
	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
	#Create saver/loader
	saver = tf.train.Saver()
	if(USE_LAST):
		print("Loading trained variables")
		if(ABSTRACTIVE):
			saver.restore(sess,"saved_models/SR_ABS.ckpt")
		else:
			saver.restore(sess,"saved_models/SR_EXT.ckpt")
	else:
		print("Initializing variables")
		sess.run(tf.global_variables_initializer())
	EPOCHS = 1

	#Function that returns precision recall in the context of ROUGE N=1
	N_rouge = 1
	precision_recall_R = rougeNScorer(1)
	#function used to display metrics
	def disp(loss,prec,rec,rou,num):
		percent = lambda  v: round(v*100/num,2)
		print(f"Loss: {loss/num}")
		print(f"Precision: {percent(prec)} %")
		print(f"Recall: {percent(rec)} %")
		print(f"ROUGE {N_rouge}: {percent(rou)}%")

	#Iterate EPCOCHS number of times
	for e in range(EPOCHS):
		print(f"Epoch {e+1}")
		#Amount of texts to use
		use = 3000
		#How many steps to do before printing average loss
		PRINT_EVERY = 50
		#How many steps to do before saving model
		SAVE_EVERY = PRINT_EVERY*4
		#Count of texts used so far
		count = 0
		#variables used to keep track of overall Epoch loss,precision, and recall
		epoch_loss = 0
		epoch_precision = 0
		epoch_recall = 0
		epoch_rouge = 0
		#variables used to keep track of cumulative loss,
		#precision, and recall.  Reset every PRINT_EVERY iterations.
		last_loss = 0
		last_precision = 0
		last_recall = 0
		last_rouge = 0
		#Iterate through data until count
		#d will be a dictionary with keys depending on Abstractive or Extractive
		#Both will have 'u_full_text' and 'u_abstract' and different others.
		#unique gfd method accounts for this.
		for d in data:
			count+=1
			print(f"Train {count}",end='\r')
			#Train using texts and additionally return loss and probabilities.
			#get feed-dict using gfd function passed into the train method call
			p,l,_ = sess.run([probabilities_tensor,loss,train_step],feed_dict=gfd(d))
			#Increment loss
			last_loss+=l
			#Generate summary
			summary = generateSummary(p,d['u_full_text'])
			#summary will be none if prediction entailed 0 length summary
			if(summary):
				precision,recall,rouge = precision_recall_R(d['u_abstract'],summary)
				#Increment precision and recall
				last_precision+=precision
				last_recall+=recall
				last_rouge+=rouge
			#Display information every PRINT_EVERY
			if(count%PRINT_EVERY==0):
				print("")
				print("_____________")
				print(f"On last {PRINT_EVERY} documents:")
				#Display average loss,preicison and recall over last PRINT_EVERY iterations
				disp(last_loss,last_precision,last_recall,last_rouge,PRINT_EVERY)
				'''
				#Display example summarization:
				print("Actual")
				print(' '.join(d['u_abstract']))
				print("Predicted")
				print(' '.join(summary))
				'''
				#If not first time print last time
				if(count>PRINT_EVERY):
					print(f"On previous {count-PRINT_EVERY} documents:")
					disp(epoch_loss,epoch_precision,epoch_recall,epoch_rouge,count-PRINT_EVERY)
				if(count%SAVE_EVERY==0):
					print("Saving Network...")
					if(ABSTRACTIVE):
						saver.save(sess, "saved_models/SR_ABS.ckpt")
					else:
						saver.save(sess, "saved_models/SR_EXT.ckpt")
				#Increment epoch loss,precision,and recall
				epoch_loss+=last_loss
				epoch_precision+=last_precision
				epoch_recall+=last_recall
				epoch_rouge+=last_rouge
				#Reset last loss,precision, and recall
				last_loss = 0
				last_recall = 0
				last_precision = 0
				last_rouge = 0
			#Break once use number of texts have been used
			if(count>=use): break

#Add additional components to graph and train abstractively
if(ABSTRACTIVE and MODE==0):

	#Number of hidden units in summary RNN
	NUM_SUMMARY_UNITS = 25

	#Number of words in summary
	num_words_summary = tf.placeholder(tf.int32)

	#Used in generating hidden states for summary words
	abstractive_training_GRU = tf.nn.rnn_cell.GRUCell(NUM_SUMMARY_UNITS)

	#Add the same linear transformation of the summary to each word embedding
	#This is equivalent to using the summary multiplied by the same weight when
	#calculating inputs for the sequential model as outlined in the paper.
	Weight_rnn_sum = weight([WORD_VECTOR_DIM,NUM_SENTENCE_UNITS*2])
	#Transposing NUM_SUMMARY_UNITS x 1 resulting vector
	summary_comp = tf.reshape(tf.matmul(Weight_rnn_sum,s),[1,WORD_VECTOR_DIM])
	#Left with num_words_summary*WORD_VECTOR dim matrix where each row is equivalent
	tiled_summary_comp = tf.tile(input=summary_comp,multiples=[num_words_summary,1])

	#Word embeddings for each word
	summary_word_embeddings = tf.placeholder(tf.float32,[None,WORD_VECTOR_DIM])

	#Add word_embeddings with summary components to create inputs for model
	#Linear transformations of the two components are now allowed.
	sequential_inputs = tf.add(summary_word_embeddings,tiled_summary_comp)

	#Hidden representations of each word from summary: num_words_summary x NUM_SUMMARY_UNITS
	hiddenStates,_ = tf.nn.dynamic_rnn(cell=abstractive_training_GRU,
									inputs=tf.reshape(sequential_inputs,[1,-1,WORD_VECTOR_DIM]),
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
	#Network done!

	print("Loading data...")
	#Load tokenized data and summary one-hot encodings
	tokenized_abstract_sentences = pickle.load(open('processedData/TokenizedAbstractSentences2007.pkl','rb'))
	tokenized_full_text_sentences = pickle.load(open('processedData/TokenizedFullTextSentences2007.pkl','rb'))
	untokenized_abstract_sentences = pickle.load(open('processedData/AbstractSentences2007.pkl','rb'))
	untokenized_full_text_sentences = pickle.load(open('processedData/FullTextSentences2007.pkl','rb'))
	summary_one_hot_sentences = pickle.load(open('processedData/summaryEncodings.pkl','rb'))

	#Aggregate between sentneces in single summaries

	#Result is num abstracts x num words x word vector dim
	tokenized_abstracts = [sum([s for s in abstract],[]) 
						   	for abstract in tokenized_abstract_sentences]
	#Result is num abstracts x num words x num vocab
	summary_one_hot_encodings = [sum([s for s in abstract],[])
						 	for abstract in summary_one_hot_sentences]
	untokenized_abstracts = [sum([s for s in abstract],[])
							for abstract in untokenized_abstract_sentences]
	#Zip together various data for easy access
	zipped_data = zip(tokenized_abstracts,
						tokenized_full_text_sentences,
						summary_one_hot_encodings,
						untokenized_abstracts,
						untokenized_full_text_sentences)
	#Change each element from a list to a dictionary to
	#be compatible with general implementation
	zipped_data = [{'abstract':a,
					'full_text':f,
					'summary_ohe':s,
					'u_abstract':u_a,
					'u_full_text':u_f}
					for a,f,s,u_a,u_f in zipped_data]

	#Go through data and return feed_dict to run graph
	def get_feed_dict(d):
		#Store necessary data
		abstract = d['abstract']
		full_text = d['full_text']
		summary_ohe = d['summary_ohe']
		#Remove 0 length sentences from full_text
		i = 0
		while(i<len(full_text)):
			if(len(full_text[i])==0):full_text.pop(i)
			i+=1
		#Initialize padded_text to the original text
		padded_text = list(full_text)
		#Mask is used to allow network to toss out filler sentences later
		mask = [True]*min(len(full_text),NUM_SENTENCES)
		#Filler sentence equivalent to sentence with one word whose word embedding is the zero vector
		#This allows for consistent and predictable behaviour on rest of netowork.
		fillerSentence = [[0]*WORD_VECTOR_DIM]
		#Pad text if necessary and complete mask with False values where fillers are present
		while(len(padded_text)<NUM_SENTENCES):
			padded_text.append(fillerSentence)
			mask.append(False)
		#Assign sentences from text to placeholders
		#Note if the there are more sentences in the text than NUM_SENTENCES those sentences will not be considered.
		fd = {placeholder:sentence for placeholder,sentence in zip(word_embeddings,padded_text)}
		#Provide values for other placeholders
		fd.update({summary_word_embeddings:abstract,
				   summary_word_indexes:summary_ohe,
				   num_words_summary:len(abstract),
				   sentences_present:mask})
		#Return feed dict
		return fd
		
	train(zipped_data,get_feed_dict)

#Implement extractive training method using probabilities_tensor and labels
if(EXTRACTIVE):
	#None represents number of sentences present
	#Should directly feed label (no padding of label)
	#label is a boolean list where 1 indicates sentence was part of 
	#extractive summary label and 0 indicates where it was not.
	extractive_label = tf.placeholder(tf.float32,[None])
	#Compute the logarithmic loss
	loss = tf.losses.log_loss(labels=extractive_label,predictions=probabilities_tensor)

	#Load required data
	print("Loading data...")
	tokenized_full_text_sentences = pickle.load(open('processedData/TokenizedFullTextSentences2007.pkl','rb'))
	untokenized_abstract_sentences = pickle.load(open('processedData/AbstractSentences2007.pkl','rb'))
	untokenized_full_text_sentences = pickle.load(open('processedData/FullTextSentences2007.pkl','rb'))
	labels = pickle.load(open('processedData/labels.pkl','rb'))

	#Concatenate sentences of each abstract
	untokenized_abstracts = [sum([s for s in abstract],[])
							for abstract in untokenized_abstract_sentences]

	#l[0] indicates label l[1] indicates the index of the labeled document
	#This is to account for the fact that some documents (~2%) do not have labels
	#Each element of list is a dictionary which stores the data by key
	zipped_data = [{'text_labels':l[1],
					'full_text': tokenized_full_text_sentences[l[0]],
					'u_abstract': untokenized_abstracts[l[0]],
					'u_full_text': untokenized_full_text_sentences[l[0]]}
					for l in labels]

	#Go through a data point and return feed_dict to run graph
	def get_feed_dict(d):
		#Get necessary components from data point
		full_text = d['full_text']
		labels = d['text_labels']
		#Remove 0 length sentences from full_text
		i = 0
		while(i<len(full_text)):
			if(len(full_text[i])==0):full_text.pop(i)
			i+=1
		#Initialize padded_text to the original text
		padded_text = list(full_text)
		#Mask is used to allow network to toss out filler sentences later
		mask = [True]*min(len(full_text),NUM_SENTENCES)
		#Filler sentence equivalent to sentence with one word whose word embedding is the zero vector
		#This allows for consistent and predictable behaviour on rest of netowork.
		fillerSentence = [[0]*WORD_VECTOR_DIM]
		#Pad text if necessary and complete mask with False values where fillers are present
		while(len(padded_text)<NUM_SENTENCES):
			padded_text.append(fillerSentence)
			mask.append(False)
		#Assign sentences from text to placeholders
		#Note if the there are more sentences in the text than NUM_SENTENCES those sentences will not be considered.
		fd = {placeholder:sentence for placeholder,sentence in zip(word_embeddings,padded_text)}
		#Provide values for other placeholders
		fd.update({extractive_label:labels,
				   sentences_present:mask})
		#Return feed dict
		return fd

	if(MODE==0):
		#Using data and feed_dict method run general train and display procedure
		train(zipped_data,get_feed_dict)
	elif(MODE==1):
		#Initialize the session
		sess = tf.Session()
		saver = tf.train.Saver()

		print("Loading trained variables")
		saver.restore(sess,"saved_models/SR_EXT.ckpt")
		
		#Takes in predicitons and label
		#Returns (true postitive rate, false positive rate)
		def ROC_Analysis(p,t):
			true_positive_count = 0
			false_positive_count = 0
			#Actual positives simply number of 1s in label
			actual_positives = sum(t)
			#Actual positives simply number of 0s in label
			actual_negatives = len(t)-actual_positives
			#Iterate over sentences
			for pred,act in zip(p,t):
				#If we predict a sentence is in summary
				if(pred):
					#If label predicts it is or not
					if(act):
						true_positive_count+=1
					else:
						false_positive_count+=1
			#Defaults to 1 if no labels were 1
			true_positive_rate = 1
			if(actual_positives): true_positive_rate = true_positive_count/actual_positives
			#Defaults to 0 if no labels were 0
			false_positive_rate = 0
			if(actual_negatives): false_positive_rate = false_positive_count/actual_negatives
			return (true_positive_rate,false_positive_rate)

		num_thresholds = 101
		thresholds = [i/100 for i in range(num_thresholds)]
		
		#First sub list Rouge metrics, then ROC
		#Rouge metrics are precision,recall,rouge score
		#ROC metrics are true positive rate, false positive rate
		metrics = [{'rouge-1':{'f':0,'p':0,'r':0},'rouge-2':{'f':0,'p':0,'r':0},
					'rouge-l':{'f':0,'p':0,'r':0},'ROC':{'tpr':0,'fpr':0}} 
					for _ in range(num_thresholds)]
		count = 0

		rouge = Rouge()
		print("Calculating metrics...")
		for d in zipped_data:
			count+=1
			print(f"Analyzing {count}",end='\r')
			p = sess.run(probabilities_tensor,feed_dict=get_feed_dict(d))

			for i,thresh in enumerate(thresholds):
				summary = generateSummary(p,d['u_full_text'],thresh)
				#summary will be none if prediction entailed 0 length summary
				if(summary):
					#metrics for this threshold
					m = metrics[i]
					#get rouge metrics
					rouge_scores = rouge.get_scores(' '.join(summary),' '.join(d['u_abstract']))[0]
					#matching key names allows for easy incremenation of rouge metrics
					for r in rouge_scores:
						for k in rouge_scores[r]:
							m[r][k]+=rouge_scores[r][k]
					#Get threshold predicitions 0 if less than thresh 1 if greater
					predictions = [int(prob>thresh) for prob in p]
					#get ROC metrics
					tpr,fpr = ROC_Analysis(predictions,d['text_labels'])
					#Increment ROC metrics
					m['ROC']['tpr']+=tpr
					m['ROC']['fpr']+=fpr
		print("")
		
		print("Calculating metric averages...")
		#Get the average for each metric by threshold
		for m in metrics:
			#for test in this set of metrics i.e. rouge-1
			for t in m:
				#for value in test i.e 'f'
				for k in m[t]:
					m[t][k]/=count

		print("Creating Rouge Curves")
		#k is either 'f', 'p', or 'r'
		#graphs scores vs threshold value
		def graphRouge(k):
			print(f"Creating {k} curve")
			x = np.array(thresholds)
			#Get metrics for every threshold
			y_1 = [m['rouge-1'][k] for m in metrics]
			y_2 = [m['rouge-2'][k] for m in metrics]
			y_l = [m['rouge-l'][k] for m in metrics]
			avg = [(y_1i+y_2i+y_li)/3 for y_1i,y_2i,y_li in zip(y_1,y_2,y_l)]
			plt.plot(x,y_1)
			plt.plot(x,y_2)
			plt.plot(x,y_l)
			plt.plot(x,avg)
			plt.plot()
			plt.legend(['Rouge-1', 'Rouge-2', 'Rouge-l','Average'], loc='upper left')
			print(f"Showing {k} curve")
			plt.show()
		#Graph precision, recall and f1
		graphRouge('p')
		graphRouge('r')
		graphRouge('f')
		max_f = max(metrics,key=lambda m:(m['rouge-1']['f']+m['rouge-2']['f']+m['rouge-l']['f'])/3)
		for i,m in enumerate(metrics):
			if(m==max_f):
				print("Best Threshold ",i)
				break

		print("Creating ROC Curve")
		ROC_Curve_x = [m['ROC']['fpr'] for m in metrics]
		ROC_Curve_y = [m['ROC']['tpr'] for m in metrics]
		#Area under the curve approximation for ROC curve
		area = 0
		for i in range(num_thresholds-1):
			avg_height = (ROC_Curve_y[i]+ROC_Curve_y[i+1])/2
			interval_length = (ROC_Curve_x[i]-ROC_Curve_x[i+1])
			area+=avg_height*interval_length
		print(f"Area under ROC Curve {area}")
		plt.scatter(ROC_Curve_x,ROC_Curve_y)
		plt.xlim(0,1)
		plt.ylim(0,1)
		print("Showing ROC Curve")
		#plt.show()
