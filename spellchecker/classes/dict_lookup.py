import nltk
from nltk.corpus import brown

try:
    nltk.find("corpora/brown")
except:
    nltk.download('brown')

class DictionaryLookUp:
    def __init__(self, corpus_txt):
        with open(corpus_txt) as c:
            self.dl_dict = set(line.strip() for line in c)
        self.freq_dict = nltk.FreqDist(word.lower() for word in brown.words())

    def get_unknown_words(self, sentence):
        unknown_words = []
        punctuation = [',', '.', '!', '\'', '$', '/', ':', ';', "-", "?", "%", "@", "#", "&", "*", "(", ")", "\""]
        first_word = True
        for word in sentence:
            if word.lower().strip("".join(punctuation)) not in self.dl_dict:
                # Checks if word is not a punctuation symbol or a number
                if word not in punctuation and not word.isdigit():
                    # Checks if the first word is a proper noun or just capitalized
                    if first_word and self.freq_dict[word] > 10:
                        unknown_words.append(word.lower())
                    # If it's not the first word and it's not capitalized, add it to unknown words
                    elif not word[0].isupper(): 
                        unknown_words.append(word.lower())
        return unknown_words