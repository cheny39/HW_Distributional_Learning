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

# dummy word that can be added to each utterance
DUMMY_WORD = 'dummy'
# dummy meaning that can be added to each setting
DUMMY_MEANING = 'DUMMY'

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

def cross_sit_learning(beta, lamda, tau):
	# parameters
	# number of possible different meanings
	# beta = 100 
	
	# before any word-meaning pair is seen, p(word|meaning) is 1/beta
	default_probability = 1/beta

	# lamda = 0.01
	# tau =  0.09

	probability = {}
	# (word,meaning): p
	association = {}
	# (word,meaning): strength

	words, word_set, meanings, meaning_set = process_input()
	
	# create a matrix all words by all meanings 460*30
	for word in word_set:
		for meaning in meaning_set:
			probability[word,meaning] = 1/beta
			association[word,meaning] = 0

	def calculate_alignment(this_word, this_meaning, setting):
	# alignment = P(w|m) / [sum for m’ in MU (P(w|m’))]
	# denominator is sum of the probabily of each meaning in the current setting when given this word
		numerator = probability[this_word, this_meaning]
		denominator = 0
		for meaning_present in setting:
			denominator += probability[this_word, meaning_present]
		return numerator/denominator

	def calculate_probability(this_word, this_meaning):
	# P(w|m) = (A(w, m) + lamda) / (sum for w’ in W (A(w’, m)) + beta * lamda )	
		numerator = association[this_word, this_meaning] + lamda
		denominator = beta * lamda
		for w, m in association:
			if m == this_meaning:
				denominator += association[w, m]
		return numerator/denominator

	# update
	for i in range(len(words)):
		sentence = words[i]+[DUMMY_WORD]
		setting = meanings[i] #+[DUMMY_MEANING]
		for word in sentence:
			for meaning in setting:
				alignment = calculate_alignment(word, meaning, setting) 
				association[word, meaning] += alignment
				probability[word, meaning] = calculate_probability(word, meaning)

	# build lexicon
	learned_lexicon = {}
	# only words in lexicon that has the strength that's above tau will be added to learned_lexicon
	for word, meaning in probability:			
		if word != DUMMY_WORD and probability[word, meaning] > tau:
			learned_lexicon[word] = meaning

	# print(probability)
	# print(association)
	# print(learned_lexicon)

	return check_against_golden(learned_lexicon)

def main():
	# parameters
	# number of possible different meanings
	beta = [100, 1000]
	lamda = 0.01
	# optimize tau
	tau = [i * 0.01 for i in range(1, 20)]
	# tau =  0.09
	for b in beta:
		for t in tau:
			print("-------- parameters (beta, lamda, tau)", b, lamda, t, "--------")
			prec, recall, f = cross_sit_learning(b, lamda, t)
			print('Result (prec, recall, f-score):', prec, recall, f)
			print()

main()

# -------- parameters (beta, lamda, tau) 1000 0.01 0.04 --------
# Result (prec, recall, f-score): 0.16455696202531644 0.38235294117647056 0.2300884955752212