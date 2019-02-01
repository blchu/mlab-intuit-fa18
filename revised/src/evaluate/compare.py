import os
import json
import matplotlib.pyplot as plt

#ROC Curve Comparison
model_names = []
plt.xlim(0,1)
plt.xlabel('False Positive Rate')
plt.ylim(0,1)
plt.ylabel('True Positive Rate')
plt.title("ROC Curves")
#For each ROC Curve saved
for file in sorted([f for f in os.listdir('Plots/ROC_Curves') if '.json' in f]):
	model_name = file[:file.index('.json')]
	model_names.append(model_name[:model_name.index('_')])
	ROC_Curve = json.load(open('Plots/ROC_Curves/'+file,'r'))
	x = ROC_Curve['x']
	y = ROC_Curve['y']
	plt.plot(x,y)
plt.legend(model_names)
plt.show()

#Document Length Bin Comparison
model_names = []
scores = {'p':[],'r':[],'f':[]}
#For each DLB Breakdown saved
for file in sorted([f for f in os.listdir('Plots/DLB_Analysis') if '.json' in f]):
	model_name = file[:file.index('.json')]
	model_names.append(model_name[:model_name.index('_')])
	DLB = json.load(open('Plots/DLB_Analysis/'+file,'r'))
	for k in scores:
		scores[k].append(DLB[k])

plt.figure(figsize=(8,6))
plt.subplot(221)
plt.title("Precision vs Document Length Bin")
for i in scores['p']: plt.plot(i)

plt.subplot(222)
plt.title("Recall vs Document Length Bin")
for i in scores['r']: plt.plot(i)

plt.subplot(223)
plt.title("F1 vs Document Length Bin")
for i in scores['f']: plt.plot(i)

plt.figlegend(labels=model_names,loc='lower right')
plt.tight_layout()
plt.show()

#Num Sentences Predicted By Label Comparison
model_names = []
scores = {'p':[],'r':[],'f':[]}
counts = []
#For each Num Predicted Breakdown saved
for file in sorted([f for f in os.listdir('Plots/NP_Analysis') if '.json' in f]):
	model_name = file[:file.index('.json')]
	model_names.append(model_name[:model_name.index('_')])
	NP = json.load(open('Plots/NP_Analysis/'+file,'r'))
	for k in scores:
		scores[k].append(NP[k])
	if(not counts): counts = NP['counts']
#1,2,...max_sent
x = range(1,len(scores['p'][0])+1)

plt.figure(figsize=(8,6))
plt.subplot(221)
plt.title("Precision vs # of Sent Pred by Label")
for i in scores['p']: plt.plot(x,i)

plt.subplot(222)
plt.title("Recall vs # of Sent Pred by Label")
for i in scores['r']: plt.plot(x,i)

plt.subplot(223)
plt.title("F1 vs # of Sent Pred by Label")
for i in scores['f']: plt.plot(x,i)

plt.subplot(224)
plt.title("Frequency of # of Sent Pred by Label")
plt.bar(x,counts)

plt.figlegend(labels=model_names,loc='center',bbox_to_anchor=(0.84,0.22))
plt.tight_layout()
plt.show()