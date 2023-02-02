import random

# dummy word that can be added to each utterance
DUMMY_WORD = 'dummy'

def process_input():
	f=open('rollins.txt')
	lines = f.read().split('\n')

	# get rid of empty lines
	utterances = [ line for line in lines if line != ""]

	# get all words
	sentences = [utterances[i] for i in range(len(utterances)) if i%2 == 0]
	words =[]
	word_set = set()
	for sentence in sentences:
		current_words = sentence.split(' ')
		words.append(current_words)
		word_set.update(current_words)
	word_set.add(DUMMY_WORD)

	# get all meanings
	settings = [utterances[i] for i in range(len(utterances)) if i%2 == 1]
	meanings = []
	meaning_set = set()
	for setting in settings:
		current_meanings = setting.split(' ')
		meanings.append(current_meanings)
		meaning_set.update(current_meanings)
	# meaning_set.add(DUMMY_MEANING)

	return words, word_set, meanings, meaning_set

def check_against_golden(learned_lexicon):
	if len(learned_lexicon) == 0:
		return None, None, None

	g = open('gold.txt')
	lines = g.read().split('\n')
	answer = {}
	for line in lines[:-1]:
		words = line.split(' ')
		answer[words[0]] = words[1]
	# print (answer)

	tp = 0
	fp = 0
	fn = 0

	for word in learned_lexicon:
		if word in answer:
			if learned_lexicon[word] == answer[word]:
				tp += 1

	for word in answer:
		if word not in learned_lexicon:
			fn += 1
	# print("tp",tp)
	# print("fp",fp)
	# print("fn",fn)
	precision = tp/len(learned_lexicon)
	recall = tp/len(answer)

	if (precision == 0 or recall == 0):
		f = None
	else:
		f = 1/(0.5*(1/precision)+0.5*(1/recall))

	return precision, recall, f


def pursuit_model(beta, gamma, lamda, theta):
	association = {}
	probability = {}

	words, word_set, meanings, meaning_set = process_input()
	
	# create a matrix all words by all meanings 460*30
	for word in word_set:
		for meaning in meaning_set:
			probability[word, meaning] = 1/beta
	
	def calculate_probability(this_word, this_meaning):
		# P(m|w) = (A(w, m) + lamda) / (sum of A(w, m') for all m' + beta * lamda )	
		numerator = association[this_word][this_meaning] + lamda
		denominator = beta * lamda
		for m in association[this_word]:
			denominator += association[this_word][m]
		return numerator/denominator


	for i in range(len(words)):
		sentence = words[i]
		setting = meanings[i] 
		for word in sentence:
			if word not in association:
				max_assoc_dict = {}
				max_assoc = 0
				for meaning in setting:
					for meaning_dict in association.values():
						if meaning in meaning_dict:
							if meaning_dict[meaning] > max_assoc:
								max_assoc = meaning_dict[meaning]
					max_assoc_dict[meaning] = max_assoc
				# print("max_assoc_dict:", max_assoc_dict)
				least_assoc = min(max_assoc_dict.values())
				least_associated_candidates = [key for key, value in max_assoc_dict.items() if value == least_assoc]
				least_associated_meaning = random.choice(least_associated_candidates)
				# print("least_associated_meaning: ",least_associated_meaning)
				association[word] = {least_associated_meaning: gamma}
				probability[word,least_associated_meaning] = calculate_probability(word,least_associated_meaning)
			else:
				# choose best hypothesized_meaning to evaluate
				# max_assoc_word = max(association[word].values())
				# best_meaning_candidates = [key for key, value in association[word].items() if value == max_assoc_word]
				# best_meaning = random.choice(best_meaning_candidates)				
				# print("best_meaning", best_meaning)
				candidates = {}
				for w,m in probability:
					if w == word:
						candidates[m] = probability[w,m]
				max_assoc_word = max(candidates.values())
				best_meaning_candidates = [key for key, value in candidates.items() if value == max_assoc_word]
				best_meaning = random.choice(best_meaning_candidates)	
				if best_meaning in setting:
					association[word][best_meaning] += gamma * (1 - association[word][best_meaning])
					probability[word,best_meaning] = calculate_probability(word,best_meaning)
				else:
					association[word][best_meaning] *= 1 - gamma
					probability[word,best_meaning] = calculate_probability(word,best_meaning)

					random_meaning = random.choice(setting)
					if random_meaning in association[word]:
						association[word][random_meaning] += gamma * (1 - association[word][random_meaning])
					else:
						association[word][random_meaning] = gamma

					probability[word,random_meaning] = calculate_probability(word,random_meaning)
	# build lexicon
	# print(dict(sorted(probability.items(), key=lambda item: item[1], reverse = True)))
	lexicon = {}		
	for word, meaning in probability:			
		if probability[word, meaning] > theta:
			lexicon[word] = meaning
	return check_against_golden(lexicon)

def main():
	# parameters
	# number of possible different meanings
	# beta = 100
	# lamda = 0.001
	# gamma = 0.02
	# # optimize theta
	# theta = [0.5+i * 0.01 for i in range(30)]
	# for t in theta:
	# 	print("-------- parameters (beta, gamma, lamda, tau)", beta, gamma, lamda, t, "--------")
	# 	prec, recall, f = pursuit_model(beta, gamma, lamda, t)
	# 	print('Result (prec, recall, f-score):', prec, recall, f)
	# 	print()

	beta = 100
	gamma = 0.05
	lamda = 0.001
	theta = 0.62
	# theta = [0.62+i * 0.01 for i in range(5)]
	# for t in theta:
	precisions = []
	recalls = []
	fs = []
	for i in range(1000):
		prec, recall, f = pursuit_model(beta, gamma, lamda, theta)
		precisions.append(prec)
		recalls.append(recall)
		fs.append(f)
	precision_ave = sum(precisions)/len(precisions)
	recall_ave = sum(recalls)/len(recalls)
	f_ave = sum(fs)/len(fs)
	print("-------- parameters (beta, gamma, lamda, tau)", beta, gamma, lamda, theta, "--------")
	print('Result (prec, recall, f-score):', precision_ave, recall_ave, f_ave)

# -------- parameters (beta, gamma, lamda, tau) 100 0.05 0.001 0.75 --------
# Result (prec, recall, f-score): 0.4666666666666667 0.20588235294117646 0.2857142857142857
# 1000 simulations (0.75)
# 0.422772106659549 0.18035294117647258 0.2517520101341268
# (0.76)
# 0.6582896048396041 0.16473529411764812 0.26209915460929534

# -------- parameters (beta, gamma, lamda, tau) 100 0.05 0.001 0.65 --------
# Result (prec, recall, f-score): 0.24581389796486028 0.28776470588235076 0.26461956971253553
# -------- parameters (beta, gamma, lamda, tau) 100 0.05 0.001 0.66 --------
# Result (prec, recall, f-score): 0.27064643251209086 0.23723529411764813 0.25222306931731464
# -------- parameters (beta, gamma, lamda, tau) 100 0.05 0.001 0.67 --------
# Result (prec, recall, f-score): 0.2811380289887347 0.23820588235294182 0.2573084380498674
# -------- parameters (beta, gamma, lamda, tau) 100 0.05 0.001 0.74 --------
# Result (prec, recall, f-score): 0.42874011967135006 0.18397058823529608 0.25647248200581074
# -------- parameters (beta, gamma, lamda, tau) 100 0.05 0.001 0.75 --------
# Result (prec, recall, f-score): 0.4271629250694263 0.1825882352941195 0.2546887870201323
# -------- parameters (beta, gamma, lamda, tau) 100 0.05 0.001 0.76 --------
# Result (prec, recall, f-score): 0.6705602675102666 0.16373529411764812 0.26170923697121606
# -------- parameters (beta, gamma, lamda, tau) 100 0.05 0.001 0.77 --------
# Result (prec, recall, f-score): 0.7130262265512258 0.16464705882353073 0.2660795008213834
main()