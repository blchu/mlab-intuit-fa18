from nltk.translate.bleu_score import sentence_bleu
from pythonrouge.pythonrouge import Pythonrouge

def bleu(lst, target_sentence, weights=(0.25, 0.25, 0.25, 0.25)):
	sent_scores_map = dict()
	max_score = None
	best_sentence = None
	best_idx = None
	
	for i, sentence in enumerate(lst):
	    sentence = [sentence]
	    score = sentence_bleu(sentence, target_sentence, weights)
	    sent_scores_map[i] = score
	    if max_score is None or score > max_score:
	        max_score = score
	        best_sentence = sentence
	        best_idx = i
	return best_idx, sent_scores_map

def untokenize(lst):
	untokenized = []
	for item in lst:
		sentence = ' '.join(item)
		untokenized.append([sentence])
	return untokenized

def rouge(lst, target_sentence, weights=None):
	# The weights parameter is currently ignored
	untokenized_list = untokenize(lst) # list of lists
	target_sentence = [target_sentence]

	sent_scores_map = dict()
	max_score = None
	best_sentence = None
	best_idx = None

	for i, sentence in enumerate(untokenized_list):
		sentence = [[sentence]]
		rouge = Pythonrouge(summary_file_exist=False,
                    summary=target_sentence, reference=sentence,
                    n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
                    recall_only=True, stemming=True, stopwords=True,
                    word_level=True, length_limit=True, length=50,
                    use_cf=False, cf=95, scoring_formula='average',
                    resampling=True, samples=1000, favor=True, p=0.5)
		score = rouge.calc_score()['ROUGE-1']
		sent_scores_map[i] = score
		if max_score is None or score > max_score:
			max_score = score
			best_sentence = sentence
			best_idx = i

	return best_idx, sent_scores_map

def f1_score(bleu_score, rouge_score):
	return float((2*(bleu_score*rouge_score))) / float((bleu_score + rouge_score))


	
	


reference = [['this', 'is', 'a', 'test']]
candidate = ['this', 'is', 'a', 'test']

print(bleu(reference, candidate))
print(rouge(reference, candidate))


# reference = [[["This is a test"], ["This is test"], ["Hello"]]]
# candidate = [["This is a test"]]

# print(bleu(reference, candidate, weights=(1, 0, 0, 0)))

# system summary(predict) & reference summary
# summary = [[" Tokyo is the one of the biggest city in the world."]]
# reference = [[["The capital of Japan, Tokyo, is the center of Japanese economy."]]]

# # initialize setting of ROUGE to eval ROUGE-1, 2, SU4
# # if you evaluate ROUGE by sentence list as above, set summary_file_exist=False
# # if recall_only=True, you can get recall scores of ROUGE
# rouge = Pythonrouge(summary_file_exist=False,
#                     summary=candidate, reference=reference,
#                     n_gram=2, ROUGE_SU4=True, ROUGE_L=False,
#                     recall_only=True, stemming=True, stopwords=True,
#                     word_level=True, length_limit=True, length=50,
#                     use_cf=False, cf=95, scoring_formula='average',
#                     resampling=True, samples=1000, favor=True, p=0.5)
# score = rouge.calc_score()
# print(score)