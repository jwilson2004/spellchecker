from nltk.metrics.distance import edit_distance

from classes.dict_lookup import DictionaryLookUp
from classes.candidate_generator import CandidateGenerator
from noisy_channel import NoisyChannelCorrector


def load_birkbeck(path="birkbeck_misspellings.dat.txt", limit=200):
    pairs = []
    correct_word = None

    with open(path, "r", encoding="utf-8", errors="ignore") as file:
        for line in file:
            line = line.strip()

            if not line:
                continue

            if line.startswith("$"):
                correct_word = line[1:].lower()
            elif correct_word:
                typo = line.lower()
                pairs.append((typo, correct_word))

            if limit and len(pairs) >= limit:
                break

    return pairs


def edit_distance_baseline(typo, candidates):
    if not candidates:
        return typo

    return min(candidates, key=lambda c: edit_distance(typo, c))


def evaluate():
    dlu = DictionaryLookUp()
    generator = CandidateGenerator(dlu.dl_dict)
    noisy = NoisyChannelCorrector()

    test_pairs = load_birkbeck(limit=20)

    results = {
        "Edit Distance": 0,
        "Noisy Channel": 0,
    }

    total = 0

    for typo, correct in test_pairs:
        print(f"Testing {total + 1}: {typo} -> {correct}")
        candidates = generator.generate_candidates(typo)

        edit_pred = edit_distance_baseline(typo, candidates)
        noisy_pred = noisy.best_candidate(typo, candidates)

        if edit_pred == correct:
            results["Edit Distance"] += 1

        if noisy_pred == correct:
            results["Noisy Channel"] += 1

        total += 1

    print("\nBirkbeck Evaluation Results")
    print("---------------------------")
    print(f"Total test examples: {total}\n")

    for method, correct_count in results.items():
        accuracy = correct_count / total
        print(f"{method}: {correct_count}/{total} = {accuracy:.2%}")


if __name__ == "__main__":
    evaluate()