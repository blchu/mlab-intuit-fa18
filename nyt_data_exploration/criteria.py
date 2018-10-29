import pickle
#Load full texts split into sentences
print("Loading tokenized texts")
full_texts = pickle.load(open('processedData/tokenizedFullTextSentences2007.pkl','rb'))
#Load untokenized (not split into sentences) texts to form ratios
print("Loading untokenized texts")
abstracts_untokenized = pickle.load(open('processedData/Abstracts2007.pkl','rb'))
full_texts_untokenized = pickle.load(open('processedData/FullTexts2007.pkl','rb'))

#List to be populated and saved containing bounds for various metrics in filtering documents
bounds = []

#Determine number of texts and indexes to select 5th, 10th, 90th, and 95th percentiles
print("Calculating percentiles")
num_texts = len(full_texts)
bottom_five = int(num_texts*0.05)
bottom_tenth = int(num_texts*0.1)
top_tenth = int(num_texts*0.9)
top_five = int(num_texts*0.95)
print(f"Number of texts: {num_texts}")
print(f"5%: {bottom_five}--10%: {bottom_tenth}--90%: {top_tenth}--95%: {top_five}")

#Establish criteria for filtering texts during parsing stage
#List with each element representing number of sentences in a document
print("Calculating full text sentence counts")
sentence_counts = sorted([len(txt) for txt in full_texts])
#Create bounds which capture central 80% of data with respect to sentence count.
min_sentence_count = sentence_counts[bottom_tenth]
max_sentence_count = sentence_counts[top_tenth]
print(f"Texts must have between {min_sentence_count} (10%) and {max_sentence_count} (90%) sentences.")
bounds.append((min_sentence_count,max_sentence_count))

#List with each element representing number of sentences in a summary
print("Calculating abstract word counts")
word_counts = sorted([len(txt) for txt in abstracts_untokenized])
#Create bounds which capture central 80% of data with respect to sentence count.
min_word_count = word_counts[bottom_tenth]
max_word_count = word_counts[top_tenth]
print(f"Abstracts must have between {min_word_count} (10%) and {max_word_count} (90%) words.")
bounds.append((min_word_count,max_word_count))

#returns average of list
avg = lambda l: sum(l)/len(l)

#List with each element representing the average sentence length of sentences in the document
print("Calculating average sentence lengths")
average_sentence_lengths = sorted([avg([len(s) for s in txt]) for txt in full_texts])
#Create bounds which capture central 90% of data with respect to average sentence length.
min_average_sentence_length = average_sentence_lengths[bottom_five]
max_average_sentence_length = average_sentence_lengths[top_five]
print(f"Average sentence length of texts must be between {min_average_sentence_length} (5%) and {max_average_sentence_length} (95%) words.")
bounds.append((min_average_sentence_length,max_average_sentence_length))

#List with each element
print("Calculating ratios")
ratios = sorted([len(a)/len(f) for a,f in zip(abstracts_untokenized,full_texts_untokenized)])
#Create bounds which capture central 80% of data with respect to ratio between abstract and full text length
min_ratio = ratios[bottom_tenth]
max_ratio = ratios[top_tenth]
print(f"Ratio between full text to abstract must be between {min_ratio} (10%) and {max_ratio} (95%)")
bounds.append((min_ratio,max_ratio))

#Save bounds
pickle.dump(bounds,open("processedData/bounds.pkl",'wb'))
