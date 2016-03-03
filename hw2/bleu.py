#!/usr/bin/env python
from __future__ import division
from itertools import islice # slicing for iterators
from collections import Counter
import pdb
import nltk
import string
from scipy.stats.mstats import gmean
import math
 
class Bleu:
	""" This class implements a sentence-level implementation of the BLEU
		MT evaluation system
	"""
	
	def __init__(self, ngram_level, beta=0.2):
		self.ngram_level = ngram_level
		self.beta = beta

	def brevity_penalty(self, h, ref):
		bp = 1
		if len(h) < len(ref):
			bp = math.exp(1 - len(ref)/len(h))
		return bp

	def ngrams(self, toks, n):
		ngrams = []
		for i in range(len(toks)-n+1):
			ngrams.append(tuple(toks[i:i+n]))
		
		return ngrams
		
	def ngram_precision(self, h, ref, n):
		refc = Counter(self.ngrams(ref, n))
		hc = Counter(self.ngrams(h, n))
		p = 0

		for wd in hc:
			if wd in refc:
				p += min(hc[wd], refc[wd])

		if sum(hc.values()) == 0:
			pdb.set_trace()
		return p/sum(hc.values())

	def ngram_precisions(self, h, ref):
		""" Returns an array of the n-gram precisions up to the ngram level
		"""
		ngram_precisions = []
		for i in xrange(1, min(self.ngram_level+1, len(ref), len(h))):
			s = self.ngram_precision(h, ref, i)
			ngram_precisions.append(s)
		return ngram_precisions
		
	def score(self, h, ref, postag=False, hpos=[], refpos=[]):
		score = 0

		if len(h) > 0:
			ngram_precisions = self.ngram_precisions(h, ref)

			bp = self.brevity_penalty(h, ref)

			if postag:
				postag_ngram_precisions = self.ngram_precisions(hpos, refpos)
				score = bp * (1-self.beta)*gmean(ngram_precisions) + \
						self.beta*gmean(postag_ngram_precisions)
			
			else:
				score = bp * gmean(ngram_precisions)

		return score

	def evaluate(self, h1, h2, ref):
		""" Scores hypothesis sentences based on the reference sentence 
			Sentences passed in as lists of strings
		"""
		h1score = 0
		h2score = 0

		if len(h1) > 0:
			h1score = self.score(h1, ref)
		if len(h2) > 0:
			h2score = self.score(h2, ref)

		if h1score > h2score:
			print -1
		elif h1score == h2score:
			print 0
		else:
			print 1
