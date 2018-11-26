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

#Number of documents from test dataset to evaluate performance on
#Set to none if want to use full dataset
NUM_DOCS = None

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

test_docs = json.load(open(data+'/data_splits.json','r'))['test']
if(not NUM_DOCS): NUM_DOCS = len(test_docs)

print(f"Analyzing model...")

print("Loading predictions...")
predictions = json.load(open(model+'/outputs/predictions.json','r'))

#data and predictions stored in one data structure
test_data = [{'text_labels':labels[d_id],
	'abstract': abstracts[d_id],
	'full_text_sentences': full_text_sentences[d_id],
	'probabilities': predictions[d_id]}
	for d_id in test_docs]


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

num_thresholds = 102
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
#Calculate metrics for which we have a non null label
for d in [dp for dp in test_data if dp['abstract']]:
	count+=1
	print(f"Analyzing {count}/{NUM_DOCS}",end='\r')
	p = d['probabilities']
	#Prediction may only consider early sentences
	#Have it predict 0 for all other sentences
	p+=[0]*(len(d['text_labels'])-len(p))

	for i,thresh in enumerate(thresholds):
		summary = generateSummary(p,d['full_text_sentences'],thresh)
		#summary will be none if prediction entailed 0 length summary
		if(summary):
			#metrics for this threshold
			m = metrics[i]
			#get rouge metrics
			rouge_scores = rouge.get_scores(summary,d['abstract'])[0]
			#matching key names allows for easy incremenation of rouge metrics
			for r in rouge_scores:
					for k in rouge_scores[r]:
						m[r][k]+=rouge_scores[r][k]
			#Get threshold predicitions 0 if less than thresh 1 if greater
			predictions = [int(prob>=thresh) for prob in p]
			#get ROC metrics
			tpr,fpr = ROC_Analysis(predictions,d['text_labels'])
			#Increment ROC metrics
			m['ROC']['tpr']+=tpr
			m['ROC']['fpr']+=fpr

	if(count==NUM_DOCS): break
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
	plt.legend(['Rouge-1', 'Rouge-2', 'Rouge-l','Average Rouge Score'], loc='upper left')
	print(f"Showing {k} curve")
	plt.xlabel('Threshold')
	plt.ylabel(k)
	plt.title(f"{k} Curves")
	plt.show()
#Graph precision, recall and f1
graphRouge('p')
graphRouge('r')
graphRouge('f')
max_f = max(metrics,key=lambda m:(m['rouge-1']['f']+m['rouge-2']['f']+m['rouge-l']['f'])/3)
for i,m in enumerate(metrics):
	if(m==max_f):
		print("Best Threshold: ",i/100)
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
plt.plot(ROC_Curve_x,ROC_Curve_y)
plt.xlim(0,1)
plt.xlabel('False Positive Rate')
plt.ylim(0,1)
plt.ylabel('True Positive Rate')
print("Showing ROC Curve")
plt.title("ROC Curve")
plt.show()