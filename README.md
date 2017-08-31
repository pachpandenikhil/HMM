# Catalan Part-of-Speech Tagger
Hidden Markov Model based part-of-speech tagger for Catalan language.

- Developed a Hidden Markov Model based part-of-speech tagger for Catalan language.
- Implemented Viterbi algorithm for decoding the most likely sequence of tags.
- Performed add-one smoothing on the transition probabilities from training and ignored the emission probabilities of unknown tokens during testing.

Core Technology: Python, JSON.

# Data
The corpus used is an adaptation from the Catalan portion of WikiCorpus v. 1.0, as follows:
- The corpus contains only a selection (< 1.2M words) from the original.
- The corpus contains only tokens and parts of speech, not lemmas and word senses.
- The part-of-speech tags have been simplified from the original, resulting in 29 tags.
- The format has been changed to the word/TAG format, with each sentence on a separate line.

# Programs

1. *hmmlearn.py*

Learns a hidden Markov model from the training data and writes the model parameters to a json file called *hmmmodel.txt*.

```
> python hmmlearn.py /path/to/train/file
```

2. *hmmdecode.py*

Reads the model parameters from the file *hmmmodel.txt*, tags each word in the test data, and writes the results to a text file called *hmmoutput.txt* in the same format as the training data.

```
> python hmmdecode.py /path/to/test/file
```

# Results
Results for tagger executed on unseen test data.

Correct: 90027

Total: 95347

Accuracy: **0.944203802951**
