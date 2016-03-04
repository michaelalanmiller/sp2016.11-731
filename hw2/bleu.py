#!/usr/bin/env python
from __future__ import division
from itertools import islice # slicing for iterators
from collections import Counter
import pdb
import nltk
import string
from scipy.stats.mstats import gmean
import math
import numpy as np
 
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

		return p/sum(hc.values())

	def ngram_precisions(self, h, ref):
		""" Returns an array of the n-gram precisions up to the ngram level
		"""
		ngram_precisions = []
		for i in xrange(1, min(self.ngram_level+1, len(ref)+1, len(h)+1)):
			s = self.ngram_precision(h, ref, i)
			ngram_precisions.append(s)
		return ngram_precisions
		
	def score(self, h, ref, postag=False, hpos=[], refpos=[], wts=[]):
		""" Weights are for ngram weights in the average
		"""
		score = 0.0

		if len(h) > 0:
			ngram_precisions = self.ngram_precisions(h, ref)
			bp = self.brevity_penalty(h, ref)

			if postag:
				postag_ngram_precisions = self.ngram_precisions(hpos, refpos)
				if wts:
					score = bp * (1-self.beta)*self.wgmean(ngram_precisions, wts) + \
						self.beta*self.wgmean(postag_ngram_precisions, wts)
				else:
					score = bp * (1-self.beta)*gmean(ngram_precisions) + \
						self.beta*gmean(postag_ngram_precisions)
			
			else:
				if wts:
					score = bp * self.wgmean(ngram_precisions, wts)
				else:
					score = bp * gmean(ngram_precisions)

		return score

	def wgmean(self, nums, weights):
		''' 
        Return the geometric average of nums
        @param    list    nums    List of nums to avg
        @return   float   Geometric avg of nums 
		'''

		numer = 0
		if any(n==0.0 for n in nums):
			return 0.0
		for i in xrange(len(nums)):
			numer += weights[i] * math.log(nums[i])
		
		return math.exp(numer/sum(weights[:len(nums)]))
