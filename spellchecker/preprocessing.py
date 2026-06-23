from nltk.tokenize.toktok import ToktokTokenizer
from nltk import word_tokenize

def get_sentence():
    while True:
        sentence = input("Please enter a string to spell check: ")
        try:
            sentence = str(sentence)
        except:
            print("Please enter a valid sentence.")
            continue
        if sentence.strip() == "":
            print("Please enter a valid sentence.")
            continue
        break
        
    return sentence

# Tokenizes the words in each review, chosen tokenizer
def toktok_split(sentence):
    toktok = ToktokTokenizer()
    # Handle the case of any possessive apostrophes or contractions
    no_apos_sent = sentence.replace("'", "<APOSTROPHE>")
    
    tokens = toktok.tokenize(no_apos_sent)
    
    final_tokens = [token.replace("<APOSTROPHE>", "'") for token in tokens]
    return final_tokens