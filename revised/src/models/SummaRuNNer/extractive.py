import numpy as np
import tensorflow as tf
import pickle
import json
import sys
from rouge import Rouge
import os
import matplotlib.pyplot as plt

#variable specificies whether to start from scratch
#or initialize using weights trained from previous run
USE_LAST = True
#Number of docs to train with (None for all)
NUM_DOCS_TRAIN = None
#Number of docs to validate with (None for all)
NUM_DOCS_VAL = 200
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
NUM_SENTENCES = json.load(open("model_specific_files/cap.json",'rb'))
print(f"NUM_SENTENCES: {NUM_SENTENCES}")
print("Setting up Graph...")
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
def generateSummary(probabilities,text,threshold=0.15):
	summary = ""
	#iterate through probabilities
	for i in range(len(probabilities)):
		#if probability>threshold append sentence to summary
		if(probabilities[i]>=threshold): summary+=text[i]
	return summary

#Train procedure
#Takes in training data and validation data
#Displays metrics on validation dataset in realtime graphs
#while training.
def train(data,validation):

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
	LEARNING_RATE = 3e-5
	print("Creating training algorithm...")
	train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
	#Create saver/loader
	saver = tf.train.Saver()
	if(USE_LAST):
		print("Loading trained variables")
		saver.restore(sess,"saved_models/SR_EXT.ckpt")
	else:
		print("Initializing variables")
		sess.run(tf.global_variables_initializer())

	#Epochs to train for
	EPOCHS = 2

	#general library rouge scorer
	rouge = Rouge()

	num_val = len([v for v in validation if v['abstract']])
	#Display Validation Performance
	#Takes in list containing past rouge scores and losses
	def dispValPerformance(r1,r2,rl,losses):
		print('')
		print("Evaluating Validation Performance")
		#Initialize validation metrics
		val_r1=0
		val_r2=0
		val_rl=0
		val_loss=0
		count=0
		#For all validation texts which we have an abstract
		for d in [dp for dp in validation if dp['abstract']]:
			count+=1
			print(f"Validating {count}/{num_val}",end='\r')
			#Computes probs and loss for text
			p,l = sess.run([probabilities_tensor,loss],feed_dict=get_feed_dict(d))
			#Ensures that at least one sentence is included in summary
			p[np.argmax(p)]=1
			#Generate summary
			summary = generateSummary(p,d['full_text_s'])
			#get rouge scores
			rouge_scores = rouge.get_scores(summary,d['abstract'])[0]
			#Increment batch metrics
			val_r1+=rouge_scores['rouge-1']['f']
			val_r2+=rouge_scores['rouge-2']['f']
			val_rl+=rouge_scores['rouge-l']['f']
			val_loss+=l
		#Normalizes metrics
		val_r1/=num_val
		val_r2/=num_val
		val_rl/=num_val
		val_loss/=num_val
		#Display metrics
		print(f"Loss: {val_loss}")
		print(f"ROUGE-1: {round(val_r1*100,2)} %")
		print(f"ROUGE-2: {round(val_r2*100,2)} %")
		print(f"ROUGE-L: {round(val_rl*100,2)}%")

		#Append to list of all past metrics
		r1.append(val_r1)
		r2.append(val_r2)
		rl.append(val_rl)
		losses.append(val_loss)

		#Close previous plots and replot for live update on metrics
		plt.close()
		#Plot rouge score trends
		plt.subplot(2,1,1)
		plt.title("Rouge Scores on Validation Set vs Time")
		plt.plot(r1)
		plt.plot(r2)
		plt.plot(rl)
		plt.legend(['Rouge 1','Rouge 2','Rouge L'])
		#Plot loss trends
		plt.subplot(2,1,2)
		plt.title("Loss on Validation Set vs Time")
		plt.plot(losses)
		plt.tight_layout()
		plt.pause(0.001)


	#Iterate EPOCHS number of times
	for e in range(EPOCHS):
		print(f"Epoch {e+1}")
		#How many steps to do before printing average loss
		PRINT_EVERY = 150
		#How many steps to do before saving model
		SAVE_EVERY = PRINT_EVERY*2
		#counter
		count = 0
		#Variables keep track of validation metrics
		r_1=[]
		r_2=[]
		r_l=[]
		losses=[]
		#Enter matplotlib interactive mode to allow for easy
		#live updates of metric graphs
		plt.ion()
		#Iterate through data where we have a non null label
		for d in [dp for dp in data if dp['abstract']]:
			#Display and graph information every PRINT_EVERY
			if(count%PRINT_EVERY==0):
				if(count%SAVE_EVERY==0):
					print("Saving Network...")
					saver.save(sess, "saved_models/SR_EXT.ckpt")
				#Print validation metrics and
				#Graph in context of previous metrics.
				dispValPerformance(r_1,r_2,r_l,losses)

			count+=1
			print(f"Train {count}",end='\r')
			train_step.run(feed_dict=get_feed_dict(d),session=sess)


def predict(data):
	
	#Initialize the session
	sess = tf.Session()
	saver = tf.train.Saver()

	print("Loading trained variables")
	saver.restore(sess,"saved_models/SR_EXT.ckpt")
	
	print("Making Summaries...")
	n = len(data)
	count = 0
	predictions = {}
	for d in data:
		count+=1
		print(f"Summarizing {count}/{n}",end='\r')
		p = sess.run(probabilities_tensor,feed_dict=get_feed_dict(d))
		#Ensures that at least one sentence is included in summary
		p[np.argmax(p)]=1
		predictions[d['doc_id']]=p.tolist()
	print("")
	
	print("Saving Summaries...")
	#Create Directory to store the predictions if it doesn't already exist
	if not os.path.exists('outputs'):
	    os.makedirs('outputs')

	json.dump(predictions,open('outputs/predictions.json','w'))
	

#None represents number of sentences present
#Should directly feed label (no padding of label)
#label is a boolean list where 1 indicates sentence was part of 
#extractive summary label and 0 indicates where it was not.
extractive_label = tf.placeholder(tf.float32,[None])
#Compute the logarithmic loss
loss = tf.losses.log_loss(labels=extractive_label,predictions=probabilities_tensor)

#Load required data
print("Loading data...")
try:
    data=sys.argv[1]
except:
    print('Please pass directory containing data')

abstracts = json.load(open(data+'/abstracts.json','r'))
unvectorized_full_texts = json.load(open(data+'/fulltexts.json','r'))
labels = json.load(open(data+'/labels.json','r'))
full_text_sentences = json.load(open(data+'/sentence_tokens.json','r'))

#Load data splits contains references to documents
#which should be used for each of 3 tasks.
data_splits = json.load(open(data+'/data_splits.json','r'))
training_docs = data_splits['train']
validation_docs = data_splits['val']
test_docs = data_splits['test']

word_vectors = pickle.load(open('model_specific_files/wv.pkl','rb'))

def get_word_vector(w):
    if(w in word_vectors): return word_vectors[w]
    return np.zeros(WORD_VECTOR_DIM)

#Go through a data point and return feed_dict to run graph
def get_feed_dict(d):
	#Get necessary components from data point
	u_full_text = d['u_full_text']
	labels = d['text_labels']
	#Remove 0 length sentences from full_text
	i = 0
	while(i<len(u_full_text)):
		if(len(u_full_text[i])==0):
			u_full_text.pop(i)
			labels.pop(i)
		else: i+=1
	if(len(u_full_text)>NUM_SENTENCES): 
		u_full_text = u_full_text[:NUM_SENTENCES]
		labels = labels[:NUM_SENTENCES]
	#Initialize padded_text to the original text vectorized
	padded_text = [[get_word_vector(w) for w in s] for s in u_full_text]
	#Mask is used to allow network to toss out filler sentences later
	mask = [True]*min(len(padded_text),NUM_SENTENCES)
	#Filler sentence equivalent to sentence with one word whose word embedding is the zero vector
	#This allows for consistent and predictable behaviour on rest of netowork.
	fillerSentence = np.array([np.zeros(WORD_VECTOR_DIM)])
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


#Depending on mode create dictionary of appropriate data
#and either train or use model
#TRAIN
if(MODE==0):
	count = 0
	zipped_data = [{'text_labels':labels[d_id],
						'abstract': abstracts[d_id],
						'u_full_text': unvectorized_full_texts[d_id],
						'full_text_s':full_text_sentences[d_id]}
						for d_id in training_docs]
	#If we want to use only a portion of the whole training set
	if(NUM_DOCS_TRAIN): zipped_data = zipped_data[:NUM_DOCS_TRAIN]
	print(f"Training with {len(zipped_data)} documents")

	validation_data = [[{'text_labels':labels[d_id],
						'abstract': abstracts[d_id],
						'u_full_text': unvectorized_full_texts[d_id],
						'full_text_s':full_text_sentences[d_id]}
						for d_id in validation_docs]]
	#If we want to use only a portion of the whole training set
	if(NUM_DOCS_VAL): validation_data = zipped_data[:NUM_DOCS_VAL]
	print(f"Validating with {len(validation_data)} documents")
	train(zipped_data,validation_data)
#USE
elif(MODE==1):
	#Make predictions for both test and validation for evaluation
	zipped_data = [{'text_labels':labels[d_id],
					'abstract': abstracts[d_id],
					'u_full_text': unvectorized_full_texts[d_id],
					'doc_id':d_id}
					for d_id in test_docs+validation_docs]
	print(f"Making predictions for {len(zipped_data)} test/validation documents")
	predict(zipped_data)


