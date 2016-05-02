# Running

```
./encoderdecoder ../data/train.src ../data/train.tgt ../data/dev.src ../data/dev.tgt ../data/test.src
```

## Output
Every time the model has a lower error than anything previous, saves to the current directory (build) the attempted translated test set as `testout_<timestamp>.txt`.

# Original text
There are two Python programs here:

 - `python bleu.py your-output.txt ref.txt` to compute the BLEU score of your output against the reference translation.
 - `python rnnlm.py ref.txt` trains an LSTM language model, just for your reference if you want to use pyCNN to perform this assignment.

The `data/` directory contains the files needed to develop the MT system:

 - `data/train.*` the source and target training files.

 - `data/dev.*` the source and target development files.

 - `data/test.src` the source side of the blind test set.
