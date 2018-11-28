import os
import json
import matplotlib.pyplot as plt

if not os.path.exists('ROC_Curves') or len(os.listdir('ROC_Curves'))==0:
	raise Exception("No ROC Curves saved")

model_names = []
plt.xlim(0,1)
plt.xlabel('False Positive Rate')
plt.ylim(0,1)
plt.ylabel('True Positive Rate')
plt.title("ROC Curves")
#For each ROC Curve saved
for file in os.listdir('ROC_Curves'):
	model_name = file[:file.index('.json')]
	model_names.append(model_name)
	ROC_Curve = json.load(open('ROC_Curves/'+file,'r'))
	x = ROC_Curve['x']
	y = ROC_Curve['y']
	plt.plot(x,y)
plt.legend(model_names)
plt.show()