import torch
import re
from transformers import AutoModelForMaskedLM, AutoTokenizer
from collections import defaultdict

def build_batch(left, right, candidate_batch, tokenizer, model) -> torch.Tensor:
    left_tokens = tokenizer.convert_tokens_to_ids(left)
    right_tokens = tokenizer.convert_tokens_to_ids(right)

    batch = []
    metadata = [] # (cand, cand_token, mask_pos)

    for tok_len in candidate_batch:
        cands = candidate_batch[tok_len]
        for i, cand in enumerate(cands):
            cand = list(cand)
            # Create the new token list with the number of mask positions matching the
            # current #tokens being considered
            for tok_id in range(len(cand)):
                masked = cand.copy()
                for j in range(tok_id, len(cand)):
                    masked[j] = tokenizer.mask_token_id
                
                # Reconstruct the token list
                input_ids = [tokenizer.cls_token_id, *left_tokens, *masked, *right_tokens, tokenizer.sep_token_id]
                batch.append(input_ids)
                metadata.append((tokenizer.decode(cand), cand[tok_id], 1 + len(left_tokens) + tok_id))

    batch_data = tokenizer.pad({"input_ids": batch}, padding=True, return_tensors="pt")
    return batch_data, metadata


def inference_with_candidates(input_sentence: str, candidates: list[set[str]], typos: list = []) -> list[dict[str, float]]:
    """
        Args: 
            input_sentence (str): sentence with misspellings as [MASK]
            candidates (list[set(str)]): List of candidates, one per misspelled position, in order (candidates[0] refers to the first [MASK])
            typo (list[str]): Optional. To display the autocorrect for this portion only
            
            Returns:
            candidates_logprobs: List of dictionaries, with each dictionary corresponding to the index of the typo. Contains
                                 log-probs from BERT for how likely each candidate word is.
                                 NOTE!!: The log-probs are not normalized. A candidate word that splits into 3 tokens gives the log-prob
                                         of (tok1 + tok2 + tok3).
        
        Set `demo` to print out corrections using just BERT alone (not very accurate if a large candidate set is given, but works decent for hand-picked examples)
    """

    demo = True

    # Trying BERT, since it has context over the entire input, not just the prefix. 
    # It might make more sense for a spell checker that has the entire text available to it already
    # Will try GPT-2 anyway later

    tokenizer = AutoTokenizer.from_pretrained(
        "google-bert/bert-base-cased", # Cased bert
    )
    
    model = AutoModelForMaskedLM.from_pretrained(
        "google-bert/bert-base-cased",
        device_map="auto",
        attn_implementation="sdpa",
    )

    if demo == True:
        if typos: assert len(typos) == len(candidates)
        print(f"\n\nInput sentence: {input_sentence}\n\n")

    # This is for the next part of the noisy channel model
    candidate_logprobs = []

    # Iteratively compute the typos so later corrections get better context
    # A little complicated because the NLTK tokens are words, while BERT's are subwords
    for typo_idx, candidate_list in enumerate(candidates):
        print(input_sentence)
        input_tokens = tokenizer.tokenize(input_sentence)
        cur_unk = input_tokens.index("[MASK]")
        left = input_tokens[0: cur_unk]
        right = input_tokens[cur_unk+1: ]

        candidate_batch = defaultdict(set) # Store by #tokens for each candidate, { n: set(candidates)}
        for token in candidate_list:
            # Get a [token_id] list from each candidate
            tokenized_candidate = tokenizer(token, return_tensors="pt").to(model.device)['input_ids'].squeeze(dim=0).tolist() # (1, T) -> (T) -> [T]
            # Length-1 tokens (BERT wraps the input in [cls] and [sep]), better to strip it
            tokenized_candidate = tuple(tokenized_candidate[1:-1])
            n_tokens = len(tokenized_candidate)
            candidate_batch[n_tokens].add(tokenized_candidate)
        
        batch, metadata = build_batch(left, right, candidate_batch, tokenizer, model)
        batch["input_ids"] = batch["input_ids"].to(model.device)
        batch["attention_mask"] = batch["attention_mask"].to(model.device)
        #for i in batch["input_ids"]:
            #print(tokenizer.decode(i))
        
        with torch.no_grad():
            outputs = model(**batch)
            predictions = outputs.logits # (B, T, V)
            prob_dists = torch.log_softmax(predictions, dim=-1)
        
        # Create the tensors to index the correct prob for each (row, token_id, pos_in_input)
        idx_tensor = torch.arange(0, len(metadata), device=model.device)
        id_tensor = torch.tensor([id for _, id, _ in metadata], device=model.device)
        pos_tensor = torch.tensor([pos for _, _, pos in metadata], device=model.device)
        #print(prob_dists.shape)
        
        candidate_probs = prob_dists[idx_tensor, pos_tensor, id_tensor].tolist() # (B) size tensor with a prob in each pos for the token
        raw_sum, n = defaultdict(float), defaultdict(int)
        for i, score in enumerate(candidate_probs):
            cand, _, _ = metadata[i]
            raw_sum[cand] += score
            n[cand] += 1
        #print(metadata)
        #print(raw_sum, n)
        norm_scores = {cand: raw_sum[cand]/n[cand] for cand in raw_sum}
        best_cand = max(raw_sum, key=raw_sum.get)
        corrected_sentence = re.sub(r"\[MASK\]", best_cand, input_sentence, count=1)
        #print(norm_scores)
        #print(input_sentence)
        candidate_logprobs.append(dict(raw_sum))

        #for i in batch["input_ids"]:
            #print(tokenizer.decode(i))

        if demo == True:
            top_n = 5 if len(norm_scores) > 5 else len(norm_scores)
            cur_typo = None if not typos else typos[typo_idx]
            print(f"Top {top_n} raw log-probs: {dict(sorted(raw_sum.items(), key=lambda p: p[1], reverse=True)[:top_n])}")
            print(f"Correction {typo_idx}: {corrected_sentence} [{cur_typo} -> {best_cand}]\n")
            if typo_idx == len(typos)-1:
                print(f"\nFinal corrected sentence: {corrected_sentence}")

        input_sentence = corrected_sentence
    #print(candidate_logprobs)
    return candidate_logprobs

if __name__ == "__main__":
    # Example sentence
    res = inference_with_candidates(input_sentence="I will have [MASK] for dinner. It's raining [MASK] and dogs!", 
                                    candidates=[{"pancakes", "stumps", "pizza", "pasta", "steak", "soup"}, {"cuts", "cats", "rhinoceroses", "bats"}], 
                                    typos=["pancaks", "cars"])
    #print(res)
