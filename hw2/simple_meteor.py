#!/usr/bin/env python
from __future__ import division
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
import pdb
import nltk
 
class SimpleMeteor:
	""" This class implements a simple version of the METEOR MT evaluation 
		system
	"""

	def __init__(self, alpha=0.5):
		self.alpha = alpha # tuning parameter for precision and recall

	def unigram_precision(self, h, ref):
		return sum(1 for w in h if w in ref)/len(h)
		
	def unigram_recall(self, h, ref):
		return sum(1 for w in h if w in ref)/len(ref)

	def score(self, h, ref):
		p = self.unigram_precision(h, ref)
		r = self.unigram_recall(h, ref)
		score = 0
		if p > 0 and r > 0:
			score = (p * r)/(self.alpha*p + (1-self.alpha)*r)

		return score

	def evaluate(self, h1, h2, ref):
		""" Scores hypothesis sentences based on the reference sentence 
			Sentences passed in as lists of strings
		"""
		h1score = self.score(h1, ref)
		h2score = self.score(h2, ref)

		if h1score > h2score:
			print -1
		elif h1score == h2score:
			print 0
		else:
			print 1

class Preprocessor:

	def preprocess(self, input, stem=False):
	# we create a generator and avoid loading all sentences into a list
		ps = nltk.PorterStemmer()

		with open(input) as f:
			for pair in f:
				alltoks = []
				for sent in pair.split(' ||| '):
					toks = nltk.word_tokenize(sent.strip().decode('utf8'))
					if stem:
						toks = [ps.stem(tok) for tok in toks]
					alltoks.append(toks)
				yield alltoks

def main():
	parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
	# PEP8: use ' and not " for strings
	parser.add_argument('-i', '--input', default='data/train-test.hyp1-hyp2-ref',
			help='input file (default data/train-test.hyp1-hyp2-ref)')
	parser.add_argument('-n', '--num_sentences', default=None, type=int,
			help='Number of hypothesis pairs to evaluate')
	# note that if x == [1, 2, 3], then x[:None] == x[:] == x (copy); no need for sys.maxint
	opts = parser.parse_args()
 
	sm = SimpleMeteor(alpha=0.505)
	p = Preprocessor()

	for h1, h2, ref in islice(p.preprocess(opts.input, stem=False), opts.num_sentences):
		sm.evaluate(h1, h2, ref)

# convention to allow import of this file as a module
if __name__ == '__main__':
	main()

