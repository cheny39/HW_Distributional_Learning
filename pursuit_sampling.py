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
	f = 1/(0.5*(1/precision)+0.5*(1/recall))

	return precision, recall, f


def pursuit_model(beta, gamma, lamda, theta):
	# beta =1000
	# gamma = 0.02
	# lamda = 0.001
	# theta =0.78
	association = {}
	probability = {}

	words, word_set, meanings, meaning_set = process_input()
	
	# create a matrix all words by all meanings 460*30
	for word in word_set:
		for meaning in meaning_set:
			probability[word,meaning] = 1/beta
	
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
				# choose hypothesized_meaning according to its probability distribution
				sampling_options = []
				sampling_distribution = []
				for w,m in probability:
					if w == word:
						sampling_options.append(m)
						sampling_distribution.append(probability[w,m])

				# best_meaning = random.choices(list(association[word].keys()), list(association[word].values()))[0]
				best_meaning = random.choices(sampling_options, weights=sampling_distribution)[0]

				if best_meaning not in association[word]:
					association[word][best_meaning] = 0

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
	# theta = [0.62+i * 0.01 for i in range(30)]
	# for t in theta:
	# 	print("-------- parameters (beta, gamma, lamda, tau)", beta, gamma, lamda, t, "--------")
	# 	prec, recall, f = pursuit_model(beta, gamma, lamda, t)
	# 	print('Result (prec, recall, f-score):', prec, recall, f)
	# 	print()

	beta = 100
	gamma = 0.05
	lamda = 0.001
	theta = 0.6
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

# -------- parameters (beta, gamma, lamda, tau) 100 0.05 0.001 0.6 --------
# Result (prec, recall, f-score): 0.3181818181818182 0.20588235294117646 0.25
main()