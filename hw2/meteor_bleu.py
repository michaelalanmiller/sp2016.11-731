#!/usr/bin/env python
from __future__ import division
import argparse # optparse is deprecated
from itertools import islice # slicing for iterators
from collections import Counter
import pdb
import nltk
import string
import numpy as np

from meteor import SimpleMeteor
from bleu import Bleu

class MeteorBleu:
	""" Prints features for all versions of Meteor and BLEU for every
		input sentence
	"""

	def __init__(self, alpha=0.5):
		self.simple_meteor = SimpleMeteor(alpha=alpha, beta=0.16)
		self.tri_bleu = Bleu(3)
		self.four_bleu = Bleu(4, beta=0.13)
		self.p = Preprocessor()

	def features(self, tokline, posline):
		""" The workhouse function
			Takes lists of tokens and postags for [h1, h2, ref]
			Returns feature values for h1, h2, h1-h2
		"""

		features = []

		# Simple Meteor
		h1p, h2p, refp = self.p.preprocess(tokline, stem=False, lowercase=False)
		h1score = self.simple_meteor.score(h1p, refp)
		h2score = self.simple_meteor.score(h2p, refp)
		h1_h2 = h1score - h2score
		features += [h1score, h2score, h1_h2]

		# Simple Meteor lowercase
		h1p, h2p, refp = self.p.preprocess(tokline, stem=False, lowercase=True)
		h1score = self.simple_meteor.score(h1p, refp)
		h2score = self.simple_meteor.score(h2p, refp)
		h1_h2 = h1score - h2score
		features += [h1score, h2score, h1_h2]

		# Simple Meteor lowercase, stemmed
		h1p, h2p, refp = self.p.preprocess(tokline, stem=True, lowercase=True)
		h1score = self.simple_meteor.score(h1p, refp)
		h2score = self.simple_meteor.score(h2p, refp)
		h1_h2 = h1score - h2score
		features += [h1score, h2score, h1_h2]

		# Simple Meteor referencing sequence of postags, lowercase, stemmed
		h1p, h2p, refp = self.p.preprocess(tokline, stem=True, lowercase=True)
		h1pos, h2pos, refpos = self.p.preprocess(posline)
		h1score = self.simple_meteor.score(
				h1p, refp, postags=True, hpos=h1pos, refpos=refpos)
		h2score = self.simple_meteor.score(
				h2p, refp, postags=True, hpos=h2pos, refpos=refpos)
		h1_h2 = h1score - h2score
		features += [h1score, h2score, h1_h2]

		# trigram BLEU, lowercased, stemmed
		h1p, h2p, refp = self.p.preprocess(tokline, stem=True, lowercase=True)
		h1score = self.tri_bleu.score(h1p, refp)
		h2score = self.tri_bleu.score(h2p, refp)
		h1_h2 = h1score - h2score
		features += [h1score, h2score, h1_h2]

		# postag-smoothed 4-gram BLEU
		h1p, h2p, refp = self.p.preprocess(tokline, stem=False, lowercase=False)
		h1pos, h2pos, refpos = self.p.preprocess(posline)
		h1score = self.four_bleu.score(
				h1p, refp, postag=True, hpos=h1pos, refpos=refpos)
		h2score = self.four_bleu.score(
				h2p, refp, postag=True, hpos=h2pos, refpos=refpos)
		h1_h2 = h1score - h2score
		features += [h1score, h2score, h1_h2]
		
		# postag-smoothed 4-gram BLEU, lowercased
		h1p, h2p, refp = self.p.preprocess(tokline, stem=False, lowercase=True)
		h1pos, h2pos, refpos = self.p.preprocess(posline)
		h1score = self.four_bleu.score(
				h1p, refp, postag=True, hpos=h1pos, refpos=refpos)
		h2score = self.four_bleu.score(
				h2p, refp, postag=True, hpos=h2pos, refpos=refpos)
		h1_h2 = h1score - h2score
		features += [h1score, h2score, h1_h2]

		# postag-smoothed 4-gram BLEU, lowercased, stemmed
		h1p, h2p, refp = self.p.preprocess(tokline, stem=True, lowercase=True)
		h1pos, h2pos, refpos = self.p.preprocess(posline)
		h1score = self.four_bleu.score(
				h1p, refp, postag=True, hpos=h1pos, refpos=refpos)
		h2score = self.four_bleu.score(
				h2p, refp, postag=True, hpos=h2pos, refpos=refpos)
		h1_h2 = h1score - h2score
		features += [h1score, h2score, h1_h2]
		
		# postag-smoothed 4-gram BLEU, lowercased, stemmed, weighted
		w = [10,5,2,1]
		h1p, h2p, refp = self.p.preprocess(tokline, stem=True, lowercase=True)
		h1pos, h2pos, refpos = self.p.preprocess(posline)
		h1score = self.four_bleu.score(
				h1p, refp, postag=True, hpos=h1pos, refpos=refpos, wts=w)
		h2score = self.four_bleu.score(
				h2p, refp, postag=True, hpos=h2pos, refpos=refpos, wts=w)
		h1_h2 = h1score - h2score
		features += [h1score, h2score, h1_h2]
		
		return features

	def evaluate(self, h1score, h2score):
		""" Scores hypothesis sentences based on scores
			Prints output
		"""
		if h1score > h2score:
			print -1
		elif h1score == h2score:
			print 0
		else:
			print 1

class Retriever:
	""" Retrieves sentences from the input file """

	def retrieve(self, input):
		""" Yields a list of lists of unpreprocessed tokens for h1, h2, ref
			Assumes the token are suffixed with _<postag>
		"""
		with open(input) as f:
			for pair in f:
				allpos = []
				alltoks = []

				for sent in pair.split(' ||| '):
					sent_wds = ''
					postags = []

					# Extract postags
					for wd in sent.strip().decode('utf8').split(' '):
						parts = wd.split('_')
						if len(parts) == 2:
							wd = parts[0]
							postag = parts[1]
						else:
							wd = '_'
							postag = parts[-1]
						sent_wds += wd + ' '
						postags.append(postag)
					
					allpos.append(postags)
					toks = nltk.word_tokenize(sent_wds.strip())
					alltoks.append(toks)

				yield alltoks, allpos

class Preprocessor:

	def preprocess(self, tokline, stem=False, lowercase=False):
		""" Takes [h1, h2, ref] as lists of unprocessed tokens
			Returns [h1, h2, ref] as lists of preprocessed tokens
			Can stem and lowercase as options, will always remove punctuation
		"""

		ps = nltk.PorterStemmer()
		ptoks = []

		for sent in tokline:
			# remove punctuation
			punct = ["''", "``", 'quot']
			punct.extend([c for c in string.punctuation])
			toks = [tok for tok in sent if not tok in punct]

			if lowercase:
				toks = [tok.lower() for tok in toks]

			if stem:
				toks = [ps.stem(tok) for tok in toks]

			ptoks.append(toks)
			
		return ptoks

def main():
	parser = argparse.ArgumentParser(description='Evaluate translation hypotheses.')
	# PEP8: use ' and not " for strings
	parser.add_argument('-i', '--input', default='data/train-test.hyp1-hyp2-ref',
			help='input file (default data/train-test.hyp1-hyp2-ref)')
	parser.add_argument('-n', '--num_sentences', default=None, type=int,
			help='Number of hypothesis pairs to evaluate')
	parser.add_argument('-o', '--output', default='meteor_bleu_features.txt.gz',
			help='output feature file (default meteor_bleu_features.txt.gz')
	opts = parser.parse_args()
 
	model = MeteorBleu(alpha=0.505)
	r = Retriever()

	allfeatures = []
	for tokline, posline in islice(r.retrieve(opts.input), opts.num_sentences):
		allfeatures.append(model.features(tokline, posline))

	featuresarr = np.asarray(allfeatures)
	np.savetxt(opts.output, featuresarr)

if __name__ == '__main__':
	main()
