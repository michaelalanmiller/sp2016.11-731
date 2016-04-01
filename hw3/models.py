#!/usr/bin/env python
# Simple translation model and language model data structures
import sys, pdb
from collections import namedtuple

# A translation model is a dictionary where keys are tuples of French words
# and values are lists of (english, logprob) named tuples. For instance,
# the French phrase "que se est" has two translations, represented like so:
# tm[('que', 'se', 'est')] = [
#     phrase(english='what has', logprob=-0.301030009985), 
#     phrase(english='what has been', logprob=-0.301030009985)]
# k is a pruning parameter: only the top k translations are kept for each f.
phrase = namedtuple("phrase", "english, logprob")
def TM(filename, k):
    sys.stderr.write("Reading translation model from %s...\n" % (filename,))
    tm = {}
    for line in open(filename).readlines():
        (f, e, logprob) = line.strip().split(" ||| ")
        tm.setdefault(tuple(f.split()), []).append(phrase(e, float(logprob)))
    for f in tm: # prune all but top k translations
        tm[f].sort(key=lambda x: -x.logprob)
        del tm[f][k:] 
    return tm

# # A language model scores sequences of English words, and must account
# # for both beginning and end of each sequence. Example API usage:
# lm = models.LM(filename)
# sentence = "This is a test ."
# lm_state = lm.begin() # initial state is always <s>
# logprob = 0.0
# for word in sentence.split():
#     (lm_state, word_logprob) = lm.score(lm_state, word)
#     logprob += word_logprob
# logprob += lm.end(lm_state) # transition to </s>, can also use lm.score(lm_state, "</s>")[1]
ngram_stats = namedtuple("ngram_stats", "logprob, backoff")
class LM:
    def __init__(self, filename):
        sys.stderr.write("Reading language model from %s...\n" % (filename,))
        self.table = {}
        for line in open(filename):
            entry = line.strip().split("\t")
            if len(entry) > 1 and entry[0] != "ngram":
                (logprob, ngram, backoff) = (float(entry[0]), tuple(entry[1].split()), float(entry[2] if len(entry)==3 else 0.0))
                self.table[ngram] = ngram_stats(logprob, backoff)

    def begin(self):
        return ("<s>",)

    def score_phrase(self, phrase):
        """ 
            Simply adds probabilities of unigram, bigram, trigrams in a phrase

            Args:
                state: tuple of strings
            Returns:
                (phrase, logprob)
        """

        unigrams = list(phrase)
        bigrams = find_ngrams(phrase, 2)
        trigrams = find_ngrams(phrase, 3)
        ngrams = unigrams + bigrams + trigrams
        score = 0.0

        for ngram in ngrams:
            if ngram in self.table:
                score += self.table[ngram].logprob
            else:
                score -= 11.0

        return (ngram, score)

    def score_phrase_dir(self, phrase):
        ngram = phrase
        score = 0.0

        # Forward
        while len(ngram)> 0: # Starts out with entire thing
            if ngram in self.table:
                #return (ngram[-2:], score + self.table[ngram].logprob)
                score += self.table[ngram].logprob
                break
            else: #backoff
                if ngram[:-1] in self.table: 
                    score += self.table[ngram[:-1]].backoff if len(ngram) > 1 else 0.0 
                    break
                ngram = ngram[1:] # Starts knocking it down

        # Backward
        ngram = phrase
        while len(ngram)> 0: # Starts out with entire thing
            if ngram in self.table:
                #return (ngram[-2:], score + self.table[ngram].logprob)
                score += self.table[ngram].logprob
                break
            else: #backoff
                if ngram[:-1] in self.table: 
                    score += self.table[ngram[:-1]].backoff if len(ngram) > 1 else 0.0 
                    break
                ngram = ngram[:-1]

        if score == 0.0:
            return (ngram, self.table[("<unk>",)].logprob)

        else:
            return (ngram, score)

    def score(self, state, word):
        """ 
            The original
            Args:
                state: tuple of strings
                word: string
            Returns:
                (states, logprob)
        """

        ngram = state + (word,)
        score = 0.0
        while len(ngram)> 0: # Starts out with entire thing
            if ngram in self.table:
                #return (ngram[-2:], score + self.table[ngram].logprob)
                return (ngram, score + self.table[ngram].logprob)
            else: #backoff
                if ngram[:-1] in self.table: 
                    score += self.table[ngram[:-1]].backoff if len(ngram) > 1 else 0.0 
                else:
                    score += 0.0
                ngram = ngram[1:] # Starts knocking it down
        return ((), score + self.table[("<unk>",)].logprob)
        
    def end(self, state):
        return self.score(state, "</s>")[1]

def find_ngrams(input_list, n):
  return zip(*[input_list[i:] for i in range(n)])
