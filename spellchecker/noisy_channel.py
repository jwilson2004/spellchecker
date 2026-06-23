"""
Noisy Channel spell correction.

This module ranks candidate corrections using the classical noisy channel idea:

    argmax_c P(c | typo) = argmax_c P(typo | c) * P(c)

P(typo | c) is estimated from the edit distance between the typo and the
candidate correction. P(c) is estimated with unigram word frequencies from the
Brown corpus, with add-one smoothing so unseen candidates still get a small
probability.
"""

import math
import nltk
from nltk.corpus import brown
from nltk.metrics.distance import edit_distance


def _load_birkbeck_words():
    import os
    possible_paths = [
        "birkbeck_misspellings.dat.txt",
        os.path.join(os.path.dirname(__file__), "birkbeck_misspellings.dat.txt"),
    ]
    words = []
    for path in possible_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    line = line.strip()
                    if line.startswith("$"):
                        words.append(line[1:].lower())
            break
    return words


class NoisyChannelCorrector:
    def __init__(self, alpha=1.0, edit_penalty=3.0):
        """
        alpha: add-one smoothing value for unigram probabilities.
        edit_penalty: larger values punish candidates farther from the typo.
        """
        self.alpha = alpha
        self.edit_penalty = edit_penalty
        try:
            self.freq_dist = nltk.FreqDist(word.lower() for word in brown.words())
        except LookupError:
            # Fallback for environments where NLTK data has not been downloaded yet.
            self.freq_dist = nltk.FreqDist(_load_birkbeck_words())

        self.vocab_size = max(len(self.freq_dist), 1)
        self.total_words = max(sum(self.freq_dist.values()), 1)

    def log_language_model_prob(self, candidate):
        """Return log P(candidate), estimated from Brown unigram counts."""
        candidate = candidate.lower()
        numerator = self.freq_dist[candidate] + self.alpha
        denominator = self.total_words + (self.alpha * self.vocab_size)
        return math.log(numerator / denominator)

    def log_edit_model_prob(self, typo, candidate):
        """
        Return log P(typo | candidate).

        We do not have a learned confusion matrix, so this uses a simple edit
        model: candidates with smaller edit distance are more likely to have
        produced the typo.
        """
        distance = edit_distance(typo.lower(), candidate.lower())
        return -self.edit_penalty * distance

    def score(self, typo, candidate):
        """Return the combined noisy channel log score for one candidate."""
        return (
            self.log_edit_model_prob(typo, candidate)
            + self.log_language_model_prob(candidate)
        )

    def rank_candidates(self, typo, candidates):
        """
        Rank candidates from best to worst.

        Returns a list of tuples: (candidate, score).
        """
        if not candidates:
            return []

        scored = []
        for candidate in candidates:
            scored.append((candidate, self.score(typo, candidate)))

        return sorted(scored, key=lambda item: item[1], reverse=True)

    def best_candidate(self, typo, candidates):
        """Return the best correction for a typo. If none exist, return typo."""
        ranked = self.rank_candidates(typo, candidates)
        if not ranked:
            return typo
        return ranked[0][0]

    def correct_unknown_words(self, unknown_words, candidate_dict):
        """
        Correct every unknown word using the noisy channel model.

        Returns:
            corrections: typo -> best correction
            ranked_candidates: typo -> list of (candidate, score)
        """
        corrections = {}
        ranked_candidates = {}

        for typo in unknown_words:
            candidates = candidate_dict.get(typo, set())
            ranked = self.rank_candidates(typo, candidates)
            ranked_candidates[typo] = ranked
            corrections[typo] = ranked[0][0] if ranked else typo

        return corrections, ranked_candidates


def apply_corrections(sentence, corrections):
    """Replace corrected words in the original sentence."""
    corrected_tokens = []
    for token in sentence.split():
        stripped = token.strip(",.!?;:\"'()[]{}")
        lower = stripped.lower()
        if lower in corrections:
            corrected_token = token.replace(stripped, corrections[lower])
            corrected_tokens.append(corrected_token)
        else:
            corrected_tokens.append(token)
    return " ".join(corrected_tokens)
