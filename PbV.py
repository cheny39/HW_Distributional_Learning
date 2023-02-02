# 1. propose but verify
import random
#build the lexicon
f=open('rollins.txt')
lines = f.read().split('\n')
# get rid of empty lines
utterances = [ line for line in lines if line != ""]
sentences = [utterances[i] for i in range(len(utterances)) if i%2 == 0]
words =[]
for sentence in sentences:
	words.append(sentence.split(' '))

settings = [utterances[i] for i in range(len(utterances)) if i%2 == 1]
meanings =[]
for setting in settings:
	meanings.append(setting.split(' '))
alpha_zero = 0
alpha = 1

NUM_ITERATION = 1000
versions = []
for _ in range(NUM_ITERATION):
	lexicon = {}

	for i in range(len(words)):
		sentence = words[i]
		setting = meanings[i]

		for word in sentence:
			if word not in lexicon:
				lexicon[word]= [random.choice(setting), alpha_zero]
			else:
				learned_meaning = lexicon[word][0]
				if learned_meaning in setting:
					lexicon[word][1] = alpha
				else:
					lexicon[word]= [random.choice(setting), alpha_zero]

	learned_lexicon = dict((k, v[0]) for k, v in lexicon.items() if v[1] == alpha)
	versions.append(learned_lexicon)
	# print(learned_lexicon)

# # check against the gold standard
g = open('gold.txt')
lines = g.read().split('\n')
answer = {}
for line in lines[:-1]:
	words = line.split(' ')
	answer[words[0]] = words[1]
# print (answer)


tp = [0]*NUM_ITERATION
fp = [0]*NUM_ITERATION
fn = [0]*NUM_ITERATION
precision = [0]*NUM_ITERATION
recall = [0]*NUM_ITERATION
f = [0]*NUM_ITERATION

for i in range(NUM_ITERATION):
	learned_lexicons = versions[i]
	for word in learned_lexicons:
		if word in answer:
			if learned_lexicons[word] == answer[word]:
				tp[i] += 1

	for word in answer:
		if word not in versions[i]:
			fn[i] += 1
	# print("tp",tp)
	# print("fp",fp)
	# print("fn",fn)
	precision[i] = tp[i]/len(learned_lexicons)
	recall[i] = tp[i]/len(answer)
	f[i] = 1/(0.5*(1/precision[i])+0.5*(1/recall[i]))

precision_ave = sum(precision)/len(precision)
recall_ave = sum(recall)/len(precision)
f_ave = sum(f)/len(f)

print(precision_ave, recall_ave, f_ave)
# 0.06193296829495655 0.3950000000000004 0.10707148014291469


