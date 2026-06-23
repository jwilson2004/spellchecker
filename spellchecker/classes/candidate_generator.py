import nltk
from nltk.metrics.distance import edit_distance

class CandidateGenerator:

    letters = 'abcdefghijklmnopqrstuvwxyz'

    def __init__(self, dict):
        self.cg_dict = dict

    def get_known_words(self, words):
        known_words = set()
        for word in words:
            if word in self.cg_dict:
                known_words.add(word)
        
        return known_words

    def edits1(self, word):
        # Get all possible splits of the word
        splits = []
        for i in range(len(word) + 1):
            splits.append((word[:i], word[i:]))
        
        deletes = []
        transposes = []
        replaces = []
        inserts = []
        for left, right in splits:
            if right:
                # Checks if deleting a single letter from the word makes a valid word
                deletes.append(left + right[1:])
                for letter in self.letters:
                    # Checks if replacing a single letter from the word makes a valid word
                    replaces.append(left + letter + right[1:])
            for letter in self.letters:
                # Checks if inserting a new letter into the word makes a valid word
                inserts.append(left + letter + right)
            if len(right) > 1:
                # Checks if switching two letters makes a valid word
                transposes.append(left + right[1] + right[0] + right[2:])

        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        edits = set()
        for edit1 in self.edits1(word):
            for edit2 in self.edits1(edit1):
                edits.add(edit2)
        
        return edits

    def generate_candidates(self, word):
        candidates = self.get_known_words(self.edits1(word)) 
        edit2_candidates = self.get_known_words(self.edits2(word))

        # If you only want edit 1 distances (or edit 2 only if there's no edit1 options),
        # comment this line out and uncomment the part below
        candidates = candidates.union(edit2_candidates)

        # if candidates:
        #     return sorted(candidates)

        # # If there's no known candidates for the typo, try depth 2
        # candidates = self.get_known_words(self.edits2(word))

        # Lambda function for sorting by the edit distance so 1 edit distance words are prioritized
        # return sorted(candidates, key=lambda x: edit_distance(word, x))
        return candidates