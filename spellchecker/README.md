**Instructions for using repository**
- Run `python3 main.py` and submit a sentence with typos to see the edit distance, LM correction, and noisy channel model solutions for the errors in the sentence.

**Code Explanation**

- `main.py`: Is the main function where the entire process is run. Gets the original sentence, tokenized sentence, the unknown words of the sentence, and all the candidates for each unknown word in the sentence. Runs the GPT-2 model and the noisy channel model.
- `preprocessing.py`: Retrieves, santizes, and tokenizes the user's sentence from the command line via `get_sentence()` and `toktok_split` (used from hw1) and currently outputs the unknown words in the sentence. 
- `classes/dict_lookup.py`: Contains the DictionaryLookUp class which stores a dictionary of all words from Hunspell's Dictionaries, specifically the large US English dictionary. It also retrieves all unknown words in the user's sentence.
- `classes/candidate_generator.py`: Uses the dictionary from DictionaryLookUp as reference, has a function to get all the known words of a sentence, uses `edits1()` as a spell checker for checking if a word is one edit distance away from being a valid word, `edits2()` is the same but for two edit distances away (uses edits1(edits1(word))). `get_candidates()` retrieves all the potential word candidates for being the fix for the misspelling, returns the set of candidates for the given word.
- `LM_correction.py`: `inference_with_candidates()` takes a sentence with typos replaced with a mask token, list of candidate words for each typo position, and optionally, the list of typos (it won't be used in determining corrections). Compute the pseudo-loglikelihood for each token(s) replacing a mask position. This is the prior for the noisy channel model. Returns a list of sets, one for each position's logprobs over that position's candidates.
- `noisy_channel.py`
Contains the `NoisyChannelCorrector` class. This file implements the Noisy Channel Model:
- `evaluation_birkbeck.py`
Loads misspellings from the Birkbeck corpus, generates correction candidates, applies the Edit Distance and Noisy Channel methods, and measures how often each method correctly recovers the intended word.


```text
best correction = argmax P(correction | misspelling)
                = argmax P(misspelling | correction) * P(correction)
```

The model combines two signals:

- Edit model: estimates `P(misspelling | correction)` using edit distance. Candidates closer to the typo receive higher scores.
- Language model prior: estimates `P(correction)` using unigram word frequencies from the Brown corpus with add-one smoothing.

The final score is computed in log space: score = log P(misspelling | correction) + log P(correction)


This gives a principled ranking that balances string similarity with how likely the correction is as an English word.

### Birkbeck Evaluation Results

We evaluated the Edit Distance Baseline and Noisy Channel Model on a 20-example sample from the Birkbeck Spelling Error Corpus.

| Method | Correct | Accuracy |
|---|---:|---:|
| Edit Distance Baseline | 3/20 | 15.00% |
| Noisy Channel Model | 5/20 | 25.00% |

The Noisy Channel Model performed better because it combines edit distance with unigram word probability, instead of only choosing the closest spelling match.


**Deliverables**

(Jay) The `birkbeck_misspellings.dat.txt` was retrieved from https://titan.dcs.bbk.ac.uk/~roger/corpora.html. We initially used the nltk.words corpus and the nltk.brown corpus, but we decided to use the Hnspell American English dictionary of words, as it contained only more common words, that way there weren't extremely uncommon words that would take the place of common errors (ae, thrys, etc.). Switching to a smaller dictionary with more common words allows for our model to be more accurate to the user's langage.



**Work Delegation**

Jay - Retrieved Birkbeck evaluation database and Hunspell Dictionary. Set up DictionaryLookUp and CandidateGenerator classes, tokenized user sentence, generated all possible typo fixes via CandidateGenerator using edit distance.

Kenneth - Implemented the LM scoring (prior for the noisy channel model). Wrote LM_correction.py using BERT to compute logprobs for each candidate to pass into the next part, and hooked it up into main.py. Also various bug fixes for other files

Jasmeen - Implemented and tested evaluation of the spell checker using the Birkbeck Spelling Error Corpus.
Created evaluation_birkbeck.py to benchmark correction performance on real-world misspellings.
Measured and compared the accuracy of the Edit Distance Baseline and Noisy Channel Model on the Birkbeck dataset.

Vanshika - Implemented the Noisy Channel Model in noisy_channel.py, combining edit-distance likelihood with unigram language model probabilities to rank candidate correctiosns. Updated main.py to run noisy-channel correction and print the best correction, top candidates, and corrected sentence.