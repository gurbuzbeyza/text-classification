import os
import re
import io
import string
import stemmer as ps
import time
import operator
from enum import Enum
import collections
import sys
from math import *
from pdb import set_trace

class Topic(Enum):
	EARN = 'earn'
	ACQ = 'acq'
	MONEY = 'money-fx'
	GRAIN = 'grain'
	CRUDE = 'crude'

class Document:
	found_topic = ''
	def __init__(self, id, topic):
		self.id = id
		self.term_freqs = {}
		self.topic = topic

bad_chars = '.:",;()\'<>'
path = 'Dataset'
stopwords = [x.strip() for x in list(open('stopwords.txt'))]
dictionary = {}
term_freqs_by_topic = {}
tokens_by_topic = {}
truth_table = {}

for name, member in Topic.__members__.items():
	term_freqs_by_topic[member] = {}
	tokens_by_topic[member] = []
	#tp #fn #fp #tn
	truth_table[member] = [0, 0, 0, 0]

training_data = [];
test_data = []


def checkIfTrainOrTest(name):
	if name == 'TRAIN':
		return 1
	elif name == 'TEST':
		return -1
	else:
		return 0

# tokenizes each document
def text_tokenizer(text, topic, isTrain, doc):
	p = ps.PorterStemmer()
	tokenized = [x.strip(bad_chars) for x in text.split() if '&#' not in x and x != '' and '&lt;' not in x]
	tokenized = [p.stem(x.lower(), 0, len(x)-1) for x in tokenized if x not in stopwords]
	terms = collections.Counter(tokenized)
	if isTrain == 1:
		tokens_by_topic[topic] += tokenized
		add_to_dict (list(terms), doc)
	return terms

# adds documents to each term
def add_to_dict(terms, doc):
	for t in terms:
		if t not in dictionary:
			dictionary[t] = [doc]
		else:
			dictionary[t].append(doc)

# checks if given topics are in the topics list
def isInTopicsList(topics):
	l = []
	for t in topics:
		for name, member in Topic.__members__.items():
			if t == member.value:
				l.append(member)
	if len(l) == 1:
		return True, l[0]
	else:
		return False, []

# returns conjunction of most discriminating words of each topics
def create_mutual_dict(first_fifty):
	mutual_dict = first_fifty[Topic.EARN].keys()
	for key, val in first_fifty.items():
		keys = set(val.keys())
		mutual_dict = mutual_dict | keys
	return mutual_dict

# Splits all the documents
def preprocessing(sgm_list):

	for s in sgm_list:
		with io.open(path + '/' + s, 'r', encoding='Latin-1') as f:
			read_data = f.read()
		reg = r'<REUTERS.*?<\/REUTERS>'
		docs_in_sgm = re.findall(reg, read_data, flags=re.DOTALL)
		for d in docs_in_sgm:
			reg = r'LEWISSPLIT=\"(.*?)\".*?NEWID=\"(\d+)\".*?<TOPICS>(.*?)</TOPICS>'
			m = re.search(reg, d, flags=re.DOTALL)
			if m:
				isTrain = checkIfTrainOrTest(m.group(1))
				id = int(m.group(2))
				topics = re.findall(r'<D>(.*?)</D>', m.group(3))
				isValid, topicName = isInTopicsList(topics)
				isValidText = True
				if isValid:
					topic = topicName
					reg = r'<TEXT>.*?<TITLE>(.*?)<\/TITLE>.*?<BODY>(.*?)<\/BODY>.*?</TEXT>'
					m = re.search(reg, d, flags=re.DOTALL)
					doc = Document(id, topic)
					if m:
						doc.term_freqs = text_tokenizer(m.group(1) + ' ' + m.group(2), topic, isTrain, doc)
					else:
						m = re.search(r'<TITLE>(.*?)<\/TITLE>', d, flags=re.DOTALL)
						if m:
							doc.term_freqs = text_tokenizer(m.group(1), topic, isTrain, doc)
						else:
							isValidText = False
					if isValidText:
						if isTrain == 1:
							training_data.append(doc)
						elif isTrain == -1:
							test_data.append(doc)

# implements naive bayes
def naive_bayes(term_freqs_of_topics, dict):

	len_of_dic = len(dict)
	num_of_docs = len(training_data)
	probs_of_topics = {}
	sum_of_topics = {}
	for name, member in Topic.__members__.items():
		probs_of_topics[member] = log(len([x for x in training_data if x.topic == member])/num_of_docs)
	probs = {}
	for doc in test_data:
		for name, member in Topic.__members__.items():
			probs[member] = probs_of_topics[member]
			sum_of_topics[member] = sum(term_freqs_of_topics[member].values())
		for key, value in doc.term_freqs.items():
			# key is term, value is frequency of that term
				for k, v in term_freqs_of_topics.items():
					# k is topic, v is term-frequncy map
					if key in dict:
						try:
							probs[k] += (log((v[key] + 1)/(sum_of_topics[k] + len_of_dic)))*value
						except:
							probs[k] += (log(1/(sum_of_topics[k] + len_of_dic)))*value
		doc.found_topic = max(probs.items(), key=operator.itemgetter(1))[0]
		if doc.topic == doc.found_topic:
			truth_table[doc.topic][0] += 1
			for name, member in Topic.__members__.items():
				if member != doc.topic:
					truth_table[member][3] += 1
		else:
			truth_table[doc.topic][1] += 1
			truth_table[doc.found_topic][2] += 1

# finds most discriminating words with mutual information technic
def mutual_information():

	docs_by_topics = {}
	mut_in = {}
	num_of_docs = len(training_data)

	for name, member in Topic.__members__.items():
		mut_in[member] = {}
		docs_by_topics[member] = [x for x in training_data if x.topic == member]
	

	for term, docs in dictionary.items():
		for name, member in Topic.__members__.items():
			inDocInClass = len(list(set(docs_by_topics[member]) & set(docs)))
			notInDocInClass = len(docs_by_topics[member]) - inDocInClass
			inDocNotInClass = len(docs) - inDocInClass
			notInDocNotInClass = (num_of_docs - len(docs_by_topics[member])) - inDocNotInClass
			total = inDocInClass + inDocNotInClass + notInDocInClass + notInDocNotInClass
			mut_in[member][term] = (inDocInClass/total)*log((inDocInClass/total)/(((inDocInClass + inDocNotInClass)/total)*(inDocInClass + notInDocInClass)/total)) if inDocInClass != 0 else 0
			mut_in[member][term] += (notInDocInClass/total)*log((notInDocInClass/total)/(((notInDocInClass + notInDocNotInClass)/total)*(inDocInClass + notInDocInClass)/total)) if notInDocInClass != 0 else 0
			mut_in[member][term] += (inDocNotInClass/total)*log((inDocNotInClass/total)/(((inDocInClass + inDocNotInClass)/total)*(inDocNotInClass + notInDocNotInClass)/total)) if inDocNotInClass != 0 else 0
			mut_in[member][term] += (notInDocNotInClass/total)*log((notInDocNotInClass/total)/(((notInDocInClass + notInDocNotInClass)/total)*(inDocNotInClass + notInDocNotInClass)/total)) if notInDocNotInClass != 0 else 0
	
	first_fifty = {}

	for name, member in Topic.__members__.items():
		first_fifty[member] = {}

	for key, value in mut_in.items():
		dict_of_topic = dict(sorted(value.items(), key=operator.itemgetter(1), reverse=True)[:50])
		for term in dict_of_topic:
			first_fifty[key][term] = term_freqs_by_topic[key][term]
	return first_fifty

#  evaluates the results
def evaluation():
	macro_precision = 0
	macro_recall = 0
	macro_F = 0
	tp = 0
	fn = 0
	fp = 0
	tn = 0
	for key, val in truth_table.items():
		macro_precision += val[0]/(val[0] + val[2])
		macro_recall += val[0]/(val[0] + val[1])
		tp += val[0]
		fn += val[1]
		fp += val[2]
		tn += val[3]
		precision = val[0]/(val[0] + val[2])
		recall = val[0]/(val[0] + val[1])
		f_value = (2*precision*recall)/(precision+recall)
		print ('Precision for ' + key.value + ': ' + str(precision))
		print ('Recall for ' + key.value + ': ' + str(recall))
		print ('F-Value for ' + key.value + ': ' + str(f_value))
		print ('----------------------')
	macro_F += (2*macro_precision*macro_recall)/(macro_precision + macro_recall)
	macro_precision /= len(Topic)
	macro_recall /= len(Topic)
	macro_F /= len(Topic)
			
	micro_precision = tp/(tp + fp)
	micro_recall = tp/(tp + fn)
	micro_F = (2*micro_precision*micro_recall)/(micro_precision + micro_recall)

	print ('Macroaveraged Precision: ' + str(macro_precision))
	print ('Macroaveraged Recall: ' + str(macro_recall))
	print ('Macroaveraged F-Value: ' + str(macro_F))
	print ('Microaveraged Precision: ' + str(micro_precision))
	print ('Microaveraged Recall: ' + str(micro_recall))
	print ('Microaveraged F-Value: ' + str(micro_F))


def __main__():

	start_time = time.time()
	arg_val = int(sys.argv[1])
	if arg_val != 0 and arg_val != 1:
		print ('Wrong argument value!')
	else:
		print ('Please wait until calculating evaluations, it may take a while.')
		sgm_list = [x for x in os.listdir(path) if x.endswith('.sgm')]
		preprocessing(sgm_list)
		for key in term_freqs_by_topic:
			term_freqs_by_topic[key] = collections.Counter(tokens_by_topic[key])
		if arg_val == 0:
			naive_bayes(term_freqs_by_topic, dictionary.keys())
		elif arg_val == 1:
			first_fifty = mutual_information()
			mutual_dict = create_mutual_dict(first_fifty)
			naive_bayes(first_fifty, list(mutual_dict))
		evaluation()

	print("--- %s seconds ---" % (time.time() - start_time))


__main__()