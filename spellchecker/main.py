import re
from classes.dict_lookup import DictionaryLookUp
from classes.candidate_generator import CandidateGenerator
from preprocessing import get_sentence, toktok_split
from LM_correction import inference_with_candidates
from noisy_channel import NoisyChannelCorrector, apply_corrections


def lm_correct(data):
    to_replace = sorted(data["unknown_words"], reverse=True)

    # Search and replace unknown words with the mask for LM scoring
    pattern = (r"".join([r"\b" + rf"{re.escape(word)}" + r"\b|" for word in to_replace])[:-1] if len(to_replace) > 1 else to_replace[0])
    masked_sentence = re.sub(pattern, "[MASK]", data["original_sentence"])
    #print(f"{data["original_sentence"]} -> {masked_sentence}")
    ordered_candidates = [set(data["candidate_dict"][unknown]) for unknown in data["unknown_words"]]
    return inference_with_candidates(masked_sentence, ordered_candidates, typos=data["unknown_words"])

def noisy_channel_correct(data):
    noisy_channel = NoisyChannelCorrector()
    corrections, ranked_candidates = noisy_channel.correct_unknown_words(
        data["unknown_words"],
        data["candidate_dict"],
    )

    print("\nNoisy Channel Model Corrections")
    for typo in data["unknown_words"]:
        print(f"{typo} -> {corrections[typo]}")
        print("Top candidates:")
        for candidate, score in ranked_candidates[typo][:5]:
            print(f"  {candidate}: {score:.4f}")

    corrected_sentence = apply_corrections(data["original_sentence"], corrections)
    print(f"\nCorrected sentence: {corrected_sentence}")

    return corrections, ranked_candidates, corrected_sentence



if __name__ == "__main__":
    corpus = "corpus/en_US-large.txt"
    dlu = DictionaryLookUp(corpus)

    sentence = get_sentence()
    tokenized_sentence = toktok_split(sentence=sentence)
    unknown_words = dlu.get_unknown_words(sentence=tokenized_sentence)
    if not unknown_words:
        print("Sentence is spelled correctly!")
    else:
        candidate_generator = CandidateGenerator(dlu.dl_dict)

        candidate_dict = {}

        for word in unknown_words:
            candidate_dict[word] = (candidate_generator.generate_candidates(word))
        
        data = {
            "original_sentence": sentence,
            "tokenized_sentence": tokenized_sentence,
            "unknown_words": unknown_words,
            "candidate_dict": candidate_dict,
        }

        for key, value in data.items():
            print(f"{key}: {value}")

        lm_correct(data)

        noisy_channel_correct(data)

        

    
