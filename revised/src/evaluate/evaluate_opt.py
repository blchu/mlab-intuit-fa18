import numpy as np
import json
from rouge import Rouge
import matplotlib.pyplot as plt
import sys
import os

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

#Number of documents you want generated/real summaries for.  These
#documents will be randomly selected each run.
NUM_PRINT = 3
#If true Full Doc will be printed along with summaries if false only
#summaries will be displayed
PRINT_FULL_DOC = False

#Load required data
print("Loading data...")
try:
    data=sys.argv[1]
    model=sys.argv[2]
    model_name = model[model.rfind('/')+1:]
except:
    raise Exception('Please pass directories containing data and model')

abstracts = json.load(open(data+'/abstracts.json','r'))
full_text_sentences = json.load(open(data+'/sentence_tokens.json','r'))
labels = json.load(open(data+'/labels.json','r'))
data_splits = json.load(open(data+'/data_splits.json','r'))

val_docs = data_splits['val']
if(not NUM_VAL_DOCS): NUM_VAL_DOCS = len(val_docs)

print(f"Analyzing {model_name}...")

print("Loading predictions...")
predictions = json.load(open(model+'/outputs/predictions.json','r'))

#data and predictions for validation dataset stored in one data structure
val_data = [{'text_labels':labels[d_id],
	'abstract': abstracts[d_id],
	'full_text_sentences': full_text_sentences[d_id],
	'probabilities': predictions[d_id]}
	for d_id in val_docs if abstracts[d_id]][:NUM_VAL_DOCS]

if os.path.exists('dl_groups.json'):
	print("Loading Document Length Group Lower Bounds...")
	dl_groups = json.load(open('dl_groups.json','r'))
else:
	#Construct list with each value representing the number of
	#sentences in a particular document and sort.
	dls = []
	for d in val_data: dls.append(len(d['full_text_sentences']))
	dls = sorted(dls)
	#Number of doc_length groups to split into.  Each group will
	#contain 10% of the document.
	num_groups = 10
	dl_groups = [dls[int(len(dls)*g/10)] for g in range(num_groups)]
	json.dump(dl_groups,open('dl_groups.json','w'))

num_groups = len(dl_groups)
print("DL Groups Lower Bounds:",dl_groups)

#accepts number of sentences and returns appropriate document length bin
def get_dl_group(length):
	if(length>dl_groups[num_groups-1]): return num_groups-1
	i = 0
	while(dl_groups[i+1]<length): i+=1
	return i


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

	count = 0

	print(f"Evaluating Thresholds on {NUM_VAL_DOCS} Validation documents...")
	for d in val_data:
		count+=1
		print(f"Analyzing {count}/{NUM_VAL_DOCS}",end='\r')
		p = d['probabilities']
		#Prediction may only consider early sentences
		#Have it predict 0 for all other sentences
		p+=[0]*(len(d['text_labels'])-len(p))

		prev_summary,prev_scores = None,None
		for i,thresh in enumerate(thresholds):
			summary = generateSummary(p,d['full_text_sentences'],thresh)
			#summary will be "" if prediction entailed 0 length summary
			if(summary):
				#Avoid redundant calculation of rouge scores if the summary
				#is no different than for the last threshold
				if(summary==prev_summary):
					rouge_scores = prev_scores
				else:
					rouge_scores = rouge.get_scores(summary,d['abstract'])[0]
				#Increment scores
				for r in ['rouge-1','rouge-2','rouge-l']:
					val_rouge[i][r]+=rouge_scores[r]['f']
				#Save summary and rouge scores
				prev_summary,prev_scores = summary,rouge_scores
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

#Rouge scores achieved by model and labels
test_rouge = {'rouge-1':{'p':0,'r':0,'f':0},'rouge-2':{'p':0,'r':0,'f':0},'rouge-l':{'p':0,'r':0,'f':0}}
label_rouge = {'rouge-1':{'p':0,'r':0,'f':0},'rouge-2':{'p':0,'r':0,'f':0},'rouge-l':{'p':0,'r':0,'f':0}}
#test and label rouge by document length
#For each document length bin find average p,r,f, aggregating the different scores
test_rouge_bl = [{'p':0,'r':0,'f':0,'count':0} for i in range(num_groups)]
label_rouge_bl = [{'p':0,'r':0,'f':0,'count':0} for i in range(num_groups)]
#test and label rouge by document length
#For each document length bin find average p,r,f, aggregating the different scores
test_rouge_ns = {}
label_rouge_ns = {}

test_ROC = [{'tpr':0,'fpr':0} for _ in range(num_thresholds)]


#cumulative word counts of sentences in summaries
model_word_count = 0
label_word_count = 0
human_word_count = 0

count=0

#Different rouge scores used
rouge_score_types = ['rouge-1','rouge-2','rouge-l']

#function used to get average scores
avg = lambda rs,t: round(sum([rs[r][t] for r in rouge_score_types])/3,2)

#select NUM_PRINT random indices to pick from 0,1,...NUM_TEST_DOCS-1
to_print = np.random.choice(NUM_TEST_DOCS,NUM_PRINT,replace=False)
print("Will print for documents",to_print)

print(f"Evalauating performance and Collecting Metrics on {NUM_TEST_DOCS} test documents...")
for d in test_data:
	count+=1
	print(f"Analyzing {count}/{NUM_TEST_DOCS}",end='\r')
	p = d['probabilities']
	#Prediction may only consider early sentences
	#Have it predict 0 for all other sentences
	p+=[0]*(len(d['text_labels'])-len(p))
	#Use only best threshold for generating summary
	summary = generateSummary(p,d['full_text_sentences'],best_threshold)
	#get document length bin for this text
	dl_group = get_dl_group(len(d['full_text_sentences']))
	#number of sentences predicted by label
	num_pred = sum(d['text_labels'])
	if(num_pred not in test_rouge_ns):
		#initialize test and label metrics for this number of sentences
		test_rouge_ns[num_pred] = {'p':0,'r':0,'f':0,'count':0}
		label_rouge_ns[num_pred] = {'p':0,'r':0,'f':0,'count':0}
	#Increment counts for aggregate statistics
	test_rouge_bl[dl_group]['count']+=1
	test_rouge_ns[num_pred]['count']+=1
	label_rouge_bl[dl_group]['count']+=1
	label_rouge_ns[num_pred]['count']+=1
	#Initialize to none in case either summary doesn't exist
	test_rouge_scores,label_rouge_scores = None,None
	#Increment model rouge metrics
	if(summary):
		test_rouge_scores = rouge.get_scores(summary,d['abstract'])[0]
		#Increment scores for each metric
		for r in rouge_score_types:
			for m in ['p','r','f']:
				test_rouge[r][m]+=test_rouge_scores[r][m]
				#Divide by 3 because 3 rouge scores and we want average
				test_rouge_bl[dl_group][m]+=test_rouge_scores[r][m]/3
				test_rouge_ns[num_pred][m]+=test_rouge_scores[r][m]/3
	#Generate label summary
	label_summary = generateSummary(d['text_labels'],d['full_text_sentences'])
	#Increment label rouge metrics
	if(label_summary):
		label_rouge_scores = rouge.get_scores(label_summary,d['abstract'])[0]
		#Increment scores for each metric
		for r in rouge_score_types:
			for m in ['p','r','f']:
				label_rouge[r][m]+=label_rouge_scores[r][m]
				#Divide by 3 because 3 rouge scores and we want average
				label_rouge_bl[dl_group][m]+=label_rouge_scores[r][m]/3
				label_rouge_ns[num_pred][m]+=label_rouge_scores[r][m]/3
	#If we want to print generated/real summary do so
	if((count-1) in to_print):
		#easier access
		lrs = label_rouge_scores
		trs = test_rouge_scores

		print("")
		print("")
		print("DOCUMENT",count)
		if(PRINT_FULL_DOC):
			print(' '.join(d['full_text_sentences']))
			print("")
		print(f"Actual Summary for Doc {count}")
		print(d['abstract'])
		print("")
		print(f"Labeled Summary p: {avg(lrs,'p')} r: {avg(lrs,'r')} f: {avg(lrs,'f')}")
		print(label_summary)
		print("")
		avg_test_rouge = sum(test_rouge_scores[k]['f'] 
			for k in rouge_score_types)/3
		print(f"Model Generated Summary p: {avg(trs,'p')} r: {avg(trs,'r')} f: {avg(trs,'f')}")
		print(summary)
		print("")
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

for i in range(num_groups):
	for m in ['p','r','f']:
		test_rouge_bl[i][m]/=test_rouge_bl[i]['count']
		label_rouge_bl[i][m]/=label_rouge_bl[i]['count']

for np in test_rouge_ns:
	for m in ['p','r','f']:
		test_rouge_ns[np][m]/=test_rouge_ns[np]['count']
		label_rouge_ns[np][m]/=label_rouge_ns[np]['count']

#Only keep range of values for ns where all values in range have
#were atleast predicted 10 times by labels
#ms will be maximum of this range
max_sent = 0
while((max_sent+1) in test_rouge_ns and test_rouge_ns[(max_sent+1)]['count']>=10): max_sent+=1
test_rouge_ns = [test_rouge_ns[i] for i in range(1,max_sent+1)]
label_rouge_ns = [label_rouge_ns[i] for i in range(1,max_sent+1)]

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
print(f"{model_name}: {model_word_count} words")
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
#Create Directory to store the ROC Curves if it doesn't already exist
if not os.path.exists('Plots/ROC_Curves'):
    os.makedirs('Plots/ROC_Curves')
#Save ROC Curve
print("Saving ROC Curve...")
ROC_Curve = {'x':ROC_Curve_x,'y':ROC_Curve_y}
json.dump(ROC_Curve,open("Plots/ROC_Curves/"+model_name+"_ROC.json",'w'))

#Display ROC Curve
print(f"Area under ROC Curve {area}")
plt.plot(ROC_Curve_x,ROC_Curve_y)
plt.xlim(0,1)
plt.xlabel('False Positive Rate')
plt.ylim(0,1)
plt.ylabel('True Positive Rate')
print("Showing ROC Curve")
plt.title(f"{model_name} ROC Curve")
plt.show()

print("Document Length Bin Analysis")
plt.figure(figsize=(8,6))
plt.subplot(221)
plt.title("Precision vs Document Length Bin")
tp = [r['p'] for r in test_rouge_bl]
lp = [r['p'] for r in label_rouge_bl]
plt.plot(range(num_groups),tp)
plt.plot(range(num_groups),lp)
plt.plot(range(num_groups),[lp[g]-tp[g] for g in range(num_groups)])

plt.subplot(222)
plt.title("Recall vs Document Length Bin")
tr = [r['r'] for r in test_rouge_bl]
lr = [r['r'] for r in label_rouge_bl]
plt.plot(range(num_groups),tr)
plt.plot(range(num_groups),lr)
plt.plot(range(num_groups),[lr[g]-tr[g] for g in range(num_groups)])

plt.subplot(223)
plt.title("F1 vs Document Length Bin")
tf = [r['f'] for r in test_rouge_bl]
lf = [r['f'] for r in label_rouge_bl]
plt.plot(range(num_groups),tf)
plt.plot(range(num_groups),lf)
plt.plot(range(num_groups),[lf[g]-tf[g] for g in range(num_groups)])

#Create Directory to store the ROC Curves if it doesn't already exist
if not os.path.exists('Plots/DLB_Analysis'):
    os.makedirs('Plots/DLB_Analysis')
#Save ROC Curve
print("Saving DLB Analysis...")
DLB = {'p':tp,'r':tr,'f':tf}
json.dump(DLB,open('Plots/DLB_Analysis/'+model_name+'_DLB.json','w'))

plt.figlegend(labels=['Model','Labels','Difference'],loc='lower right')
plt.tight_layout()
plt.show()

print("Number of Sentences Predicted by Label Analysis")
plt.figure(figsize=(8,6))
plt.subplot(221)
plt.title("Precision vs # of Sent Pred by Label")
tp = [r['p'] for r in test_rouge_ns]
lp = [r['p'] for r in label_rouge_ns]
plt.plot(range(1,max_sent+1),tp)
plt.plot(range(1,max_sent+1),lp)
plt.plot(range(1,max_sent+1),[lp[g]-tp[g] for g in range(max_sent)])

plt.subplot(222)
plt.title("Recall vs # of Sent Pred by Label")
tr = [r['r'] for r in test_rouge_ns]
lr = [r['r'] for r in label_rouge_ns]
plt.plot(range(1,max_sent+1),tr)
plt.plot(range(1,max_sent+1),lr)
plt.plot(range(1,max_sent+1),[lr[g]-tr[g] for g in range(max_sent)])

plt.subplot(223)
plt.title("F1 vs # of Sent Pred by Label")
tf = [r['f'] for r in test_rouge_ns]
lf = [r['f'] for r in label_rouge_ns]
plt.plot(range(1,max_sent+1),tf)
plt.plot(range(1,max_sent+1),lf)
plt.plot(range(1,max_sent+1),[lf[g]-tf[g] for g in range(max_sent)])

#Create Directory to store the ROC Curves if it doesn't already exist
if not os.path.exists('Plots/NP_Analysis'):
    os.makedirs('Plots/NP_Analysis')
#Save ROC Curve
print("Saving NP Analysis...")
DLB = {'p':tp,'r':tr,'f':tf,'counts':[g['count'] for g in test_rouge_ns]}
json.dump(DLB,open('Plots/NP_Analysis/'+model_name+'_NP.json','w'))

plt.figlegend(labels=['Model','Labels','Difference'],loc='lower right')
plt.tight_layout()
plt.show()

