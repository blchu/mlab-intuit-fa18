import numpy as np
import json
from rouge import Rouge
import matplotlib.pyplot as plt
import sys

'''
Evaluate a model by running evaluate.py <data folder> <model folder>
<model folder> should have a folder titled 'outputs' and inside that
folder probability predictions titled 'predictions.json'
'''
#Number of validation docs to use to determine optimal threshold
#Set to none if want to use full validation set
NUM_VAL_DOCS = None

#Number of documents from test dataset to evaluate performance on
#Set to none if want to use full test set
NUM_TEST_DOCS = None

#Load required data
print("Loading data...")
try:
    data=sys.argv[1]
    model=sys.argv[2]
except:
    print('Please pass directory containing data')

abstracts = json.load(open(data+'/abstracts.json','r'))
full_text_sentences = json.load(open(data+'/sentence_tokens.json','r'))
labels = json.load(open(data+'/labels.json','r'))
data_splits = json.load(open(data+'/data_splits.json','r'))

val_docs = data_splits['val']
if(not NUM_VAL_DOCS): NUM_VAL_DOCS = len(val_docs)

print(f"Analyzing model...")

print("Loading predictions...")
predictions = json.load(open(model+'/outputs/predictions.json','r'))


#Takes in probabilities for each sentence and corresponding text
#Returns a string containing each sentence for which the corresponding
#probability is greater than 0.5
def generateSummary(probabilities,text,threshold=0.5):
	summary = ""
	#iterate through probabilities
	for i in range(len(probabilities)):
		#if probability>threshold append sentence to summary
		if(probabilities[i]>=threshold): summary+=text[i]
	return summary

#Library rouge scorer
rouge = Rouge()

num_thresholds = 102
thresholds = [i/100 for i in range(num_thresholds)]

best_threshold = None
try:
	best_threshold=float(sys.argv[3])
	print(f"Threshold {best_threshold} provided")
except:
	print("No threshold provided.  Proceeding with Threshold Optimization...")

#If threshold not provided 
if(not best_threshold):
	#Stores rouge scores for each threshold on each document
	val_rouge = [{'rouge-1':0,'rouge-2':0,'rouge-l':0} for _ in range(num_thresholds)]

	#data and predictions for validation dataset stored in one data structure
	val_data = [{'text_labels':labels[d_id],
		'abstract': abstracts[d_id],
		'full_text_sentences': full_text_sentences[d_id],
		'probabilities': predictions[d_id]}
		for d_id in val_docs if abstracts[d_id]][:NUM_VAL_DOCS]


	count = 0

	print(f"Evaluating Thresholds on {NUM_VAL_DOCS} Validation documents...")
	for d in val_data:
		count+=1
		print(f"Analyzing {count}/{NUM_VAL_DOCS}",end='\r')
		p = d['probabilities']
		#Prediction may only consider early sentences
		#Have it predict 0 for all other sentences
		p+=[0]*(len(d['text_labels'])-len(p))

		for i,thresh in enumerate(thresholds):
			summary = generateSummary(p,d['full_text_sentences'],thresh)
			#summary will be "" if prediction entailed 0 length summary
			if(summary):
				rouge_scores = rouge.get_scores(summary,d['abstract'])[0]
				#Increment scores
				for r in ['rouge-1','rouge-2','rouge-l']:
					val_rouge[i][r]+=rouge_scores[r]['f']
	print()
	print()
	print("Finding Optimal Threshold based on Validation Evaluation...")
	best_threshold = None
	#Maximizes combined rouge scores
	max_f = max(val_rouge,key=lambda m:(m['rouge-1']+m['rouge-2']+m['rouge-l']))
	for i,m in enumerate(val_rouge):
		if(m==max_f):
			best_threshold = i/100
			break
	print("Optimal Model Threshold:",best_threshold)
	print()

test_docs = data_splits['test']
if(not NUM_TEST_DOCS): NUM_TEST_DOCS = len(test_docs)

#data and predictions for test dataset stored in one data structure
test_data = [{'text_labels':labels[d_id],
	'abstract': abstracts[d_id],
	'full_text_sentences': full_text_sentences[d_id],
	'probabilities': predictions[d_id]}
	for d_id in test_docs if abstracts[d_id]][:NUM_TEST_DOCS]

#Takes in predicitons and label
#Returns (true postitive rate, false positive rate)
def ROC_Analysis(p,t):
	true_positive_count = 0
	false_positive_count = 0
	#Actual positives simply number of 1s in label
	actual_positives = sum(t)
	#Actual negatives simply number of 0s in label
	actual_negatives = len(t)-actual_positives
	#Iterate over sentences
	for pred,act in zip(p,t):
		#If we predict a sentence is in summary\
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
	false_positive_rate = 1
	if(actual_negatives): false_positive_rate = false_positive_count/actual_negatives
	return (true_positive_rate,false_positive_rate)

test_rouge = {'rouge-1':{'p':0,'r':0,'f':0},'rouge-2':{'p':0,'r':0,'f':0},'rouge-l':{'p':0,'r':0,'f':0}}
label_rouge = {'rouge-1':{'p':0,'r':0,'f':0},'rouge-2':{'p':0,'r':0,'f':0},'rouge-l':{'p':0,'r':0,'f':0}}

test_ROC = [{'tpr':0,'fpr':0} for _ in range(num_thresholds)]


#cumulative word counts of sentences in summaries
model_word_count = 0
label_word_count = 0
human_word_count = 0

count=0

print(f"Evalauating performance on {NUM_TEST_DOCS} test documents...")
for d in test_data:
	count+=1
	print(f"Analyzing {count}/{NUM_TEST_DOCS}",end='\r')
	p = d['probabilities']
	#Prediction may only consider early sentences
	#Have it predict 0 for all other sentences
	p+=[0]*(len(d['text_labels'])-len(p))
	#Use only best threshold for generating summary
	summary = generateSummary(p,d['full_text_sentences'],best_threshold)
	#Increment model rouge metrics
	if(summary):
		rouge_scores = rouge.get_scores(summary,d['abstract'])[0]
		#Increment scores for each metric
		for r in ['rouge-1','rouge-2','rouge-l']:
			for m in ['p','r','f']:
				test_rouge[r][m]+=rouge_scores[r][m]
	#Generate label summary
	label_summary = generateSummary(d['text_labels'],d['full_text_sentences'])
	#Increment label rouge metrics
	if(label_summary):
		rouge_scores = rouge.get_scores(label_summary,d['abstract'])[0]
		#Increment scores for each metric
		for r in ['rouge-1','rouge-2','rouge-l']:
			for m in ['p','r','f']:
				label_rouge[r][m]+=rouge_scores[r][m]
	#increment word count metrics
	#word counts are the number of spaces in string + 1
	model_word_count+=summary.count(' ')+1
	label_word_count+=label_summary.count(' ')+1
	human_word_count+=d['abstract'].count(' ')+1
	#Get ROC Metrics
	for i,thresh in enumerate(thresholds):
		#Get threshold predicitions 0 if less than thresh 1 if greater
		predictions = [int(prob>=thresh) for prob in p]
		#get ROC metrics
		tpr,fpr = ROC_Analysis(predictions,d['text_labels'])
		#Increment ROC metrics
		test_ROC[i]['tpr']+=tpr
		test_ROC[i]['fpr']+=fpr

#Normalize metrics

for r in ['rouge-1','rouge-2','rouge-l']:
	for m in ['p','r','f']:
		test_rouge[r][m]/=NUM_TEST_DOCS
		label_rouge[r][m]/=NUM_TEST_DOCS


for t in test_ROC:
	t['tpr']/=NUM_TEST_DOCS
	t['fpr']/=NUM_TEST_DOCS

model_word_count/=NUM_TEST_DOCS
label_word_count/=NUM_TEST_DOCS
human_word_count/=NUM_TEST_DOCS

print()
print("ROUGE Comparison (Model Performance vs Label Performance)")
#Function used to get in percent format 0.123456 -> 12.35
p = lambda v: round(v*100,2)
for r in ['rouge-1','rouge-2','rouge-l']:
	print()
	print(r)
	#model scores
	ms = test_rouge[r]
	#label scores
	ls = label_rouge[r]
	print(f"Precision: {p(ms['p'])} vs {p(ls['p'])}")
	print(f"Recall:    {p(ms['r'])} vs {p(ls['r'])}")
	print(f"F1 score:  {p(ms['f'])} vs {p(ls['f'])}")

print()
print("Average Word Counts")
print(f"Model: {model_word_count} words")
print(f"Label: {label_word_count} words")
print(f"Human: {human_word_count} words")
print()
print("Creating ROC Curve")
ROC_Curve_x = [m['fpr'] for m in test_ROC]
ROC_Curve_y = [m['tpr'] for m in test_ROC]
#Area under the curve approximation for ROC curve
area = 0
for i in range(num_thresholds-1):
	avg_height = (ROC_Curve_y[i]+ROC_Curve_y[i+1])/2
	interval_length = (ROC_Curve_x[i]-ROC_Curve_x[i+1])
	area+=avg_height*interval_length
print(f"Area under ROC Curve {area}")
plt.plot(ROC_Curve_x,ROC_Curve_y)
plt.xlim(0,1)
plt.xlabel('False Positive Rate')
plt.ylim(0,1)
plt.ylabel('True Positive Rate')
print("Showing ROC Curve")
plt.title("ROC Curve")
plt.show()