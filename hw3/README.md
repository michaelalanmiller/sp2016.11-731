# Overview
This code contains a baseline decoder and a few attempts to improve on it, none of them successful.

The baseline is at '/.baseline'.

# Attempts
## Reordering
Following [Collins et al, 2005](http://acl.ldc.upenn.edu/P/P05/P05-1066.pdf), we reordered nodes in source-side constituency parses.
We swapped nodes for adjectives and nouns in NPs, but the results were worse than the baseline.
Our theory about this is that reordering made us miss opportunities to use the translation model's knowledge about certain phrases. 

## Retranslating
We borrowed the RETRANS Gibbs operators from [Arun et al, 2009](http://www.aclweb.org/anthology/W09-1114). 
After getting the best sentence hypothesis from the baseline, we attempted to retranslate each phrase, choosing the best possibility with a combination of translation model probability and language model probability of the phrase withe the preceding and following phrase.
Perhaps because our calculation for language model probability of this phrase wasn't very good, this also didn't improve the baseline.

# References
* Arun, A., Blunsom, P., Dyer, C., Lopez, A., Haddow, B., & Koehn, P. (2009). Monte Carlo inference and maximization for phrase-based translation. Computational Linguistics, 23(4), 102 – 110. http://doi.org/10.3115/1596374.1596394 

* Collins, M., Koehn, P., & Kučerová, I. (2005). Clause restructuring for statistical machine translation. Proceedings of the 43rd Annual Meeting on Association for Computational Linguistics, (June), 531–540. http://doi.org/10.3115/1219840.1219906

# Original text
There are three Python programs here (`-h` for usage):

 - `./decode` a simple non-reordering (monotone) phrase-based decoder
 - `./grade` computes the model score of your output

The commands are designed to work in a pipeline. For instance, this is a valid invocation:

    ./decode | ./grade


The `data/` directory contains the input set to be decoded and the models

 - `data/input` is the input text

 - `data/lm` is the ARPA-format 3-gram language model

 - `data/tm` is the phrase translation model

