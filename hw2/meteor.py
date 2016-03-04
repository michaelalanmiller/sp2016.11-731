#!/usr/bin/env python
from __future__ import division
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from collections import Counter
import pdb
import nltk
import string
 
class SimpleMeteor:
	""" This class implements a simple version of the METEOR MT evaluation 
		system
	"""

	def __init__(self, alpha=0.5, beta=0.13):
		self.alpha = alpha # tuning parameter for precision and recall
		self.beta = beta # tuning parameter for tokens and postags

	def unigram_precision(self, h, ref):
		refc = Counter(ref)
		hc = Counter(h)
		p = 0

		for wd in hc:
			if wd in refc:
				p += min(hc[wd], refc[wd])

		if len(h) > 0:
			return p/len(h)
		else:
			return 0.0
		
	def unigram_recall(self, h, ref):
		refc = Counter(ref)
		hc = Counter(h)
		r = 0

		for wd in hc:
			if wd in refc:
				r += min(hc[wd], refc[wd])
		if len(ref) > 0:
			return r/len(ref)
		else:
			return 0.0

	def score(self, h, ref, postags=False, hpos=[], refpos=[]):
		p = self.unigram_precision(h, ref)
		r = self.unigram_recall(h, ref)
		score = 0
		if p > 0 and r > 0:
			score = (p * r)/(self.alpha*p + (1-self.alpha)*r)
			if postags:
				ppos = self.unigram_precision(hpos, refpos)
				rpos = self.unigram_recall(hpos, refpos)
				if ppos > 0 and rpos > 0:
					score = (1-self.beta)*score + \
						(self.beta * ((ppos * rpos)/(self.alpha*ppos + (1-self.alpha)*rpos)))
				else:
					score = (1-self.beta)*score

		return score

class Preprocessor:

	def preprocess(self, input, stem=False):
	# we create a generator and avoid loading all sentences into a list
		ps = nltk.PorterStemmer()

		with open(input) as f:
			for pair in f:
				alltoks = []
				for sent in pair.split(' ||| '):
					# tokenize
					toks = nltk.word_tokenize(sent.strip().decode('utf8'))
		
					# remove punctuation
					punct = ["''", "``", 'quot']
					punct.extend([c for c in string.punctuation])
					toks = [tok for tok in toks if not tok in punct]

					# stem
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

	for h1, h2, ref in islice(p.preprocess(opts.input, stem=True), opts.num_sentences):
		sm.evaluate(h1, h2, ref)

# convention to allow import of this file as a module
if __name__ == '__main__':
	main()
