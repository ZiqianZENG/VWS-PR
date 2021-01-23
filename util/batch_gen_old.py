import numpy as np
import random
from tqdm import tqdm
from math import ceil

def create_one_batch(arg, ids, corpus, need_neg_senti=False):
	x, y, senti, weight, senti_words = corpus

	batch_x = []
	max_len = 0
	for i in ids:
		doc = x[i]
		if len(x[i]) > arg.max_len:
			doc = doc[:arg.max_len]
		else:
			doc = doc + [0] * (arg.max_len - len(doc))
		batch_x.append(doc)

	batch_x = np.asarray( batch_x, dtype=np.float32)

	batch_y = np.asarray([np.eye(arg.score_scale)[y[i]] if y[i] >= 0 else np.zeros(
		arg.score_scale) for i in ids], dtype=np.float32)

	batch_senti = []
	batch_weight = []
	batch_neg_senti = [] 

	for i in ids:
		neg_senti_ = []
		senti_count = senti_words.count
		words = []
		p = []

		for idx, word in enumerate(senti[i]):
			if word in senti_count:
				words.append(idx)
				p.append(senti_count[word] ** -0.25)

		if len(p) > 0:
			total = sum(p)
			p = [k / total for k in p]
			ran_val = np.random.choice(words, arg.num_senti, p=p)
			senti_ = [senti[i][val] for val in ran_val]
			weight_ = [weight[i][val] for val in ran_val]
			if need_neg_senti:
				neg_senti_ = []
				for _ in range(arg.num_neg):
					rand_senti = senti_[0]
					while rand_senti in senti_:
						rand_senti = senti_words.sample(min_count=arg.min_count)
					neg_senti_.append(rand_senti)
		else:
			# a review has no extracted tuples
			senti_ = [0 for _ in range(arg.num_senti)]
			weight_ = [0. for _ in range(arg.num_senti)]
			if need_neg_senti:
				neg_senti_ = [0 for _ in range(arg.num_neg)]

		batch_senti.append(senti_)
		batch_weight.append(weight_)
		batch_neg_senti.append(neg_senti_)
	
	batch_senti = np.asarray(batch_senti, dtype=np.int32)
	batch_weight = np.asarray(batch_weight, dtype=np.float32)
	batch_neg_senti = np.asarray(batch_neg_senti, dtype=np.int32)

	return batch_x, batch_y, batch_senti, batch_neg_senti, batch_weight


def batch_generator(arg, shuffle, corpus, need_neg_senti=False):
	batch_size = arg.batch_size
	num_docs = len(corpus[0])
	idxs = list(range(num_docs))
	if shuffle:
		random.shuffle(idxs)
	for i in tqdm( range(ceil(num_docs/batch_size)) ):
		batch_idx = idxs[i*batch_size:(i+1)*batch_size]
		if len(batch_idx) < batch_size:
			batch_idx = batch_idx + idxs[-(batch_size-len(batch_idx)):]
		batch_x, batch_y, batch_senti, batch_neg_senti, batch_weight = create_one_batch(
			arg, batch_idx, corpus,need_neg_senti)
		yield batch_x, batch_y, batch_senti, batch_neg_senti, batch_weight


def list_wrapper(lis):
	def tmp():
		for i in lis:
			yield i
	return tmp