
from collections import Counter
import os
import random


def get_clean_words(reprocess_raw: bool = False) -> list[str]:
    if reprocess_raw or not os.path.exists("clean_bee_movie_script.txt"):
        # Clean up the raw data
        with open("bee_movie_script.txt", "r") as f:
            chars_to_remove = ",;\""
            sentence_breaks = ".?!"
            raw = f.read().upper().strip()
            for ch in sentence_breaks:
                raw = raw.replace(ch, f"\n")
            sentences = raw.split("\n")
            clean_translation = str.maketrans("", "", chars_to_remove)
            cleaned_sentences = []
            for s in sentences:
                s = s.translate(clean_translation).strip()
                if s:
                    cleaned_sentences.append(f"<s> {s} </s>")

            cleaned = "\n".join(cleaned_sentences)
            with open("clean_bee_movie_script.txt", "w") as g:
                g.write(cleaned)
            clean_list = cleaned.split()
    else:
        with open("clean_bee_movie_script.txt", "r") as f:
            clean_list = f.read().split()
    
    return clean_list

def build_ngram_counts(words: list[str], max_n: int) -> list[Counter[tuple[str, ...]]]:
    """Build n-gram counts for n = 1..max_n"""
    ngram_counts = [Counter() for _ in range(max_n)]

    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            ngram = tuple(words[i : i + n])
            ngram_counts[n - 1][ngram] += 1

    return ngram_counts

def save_ngram_counts(ngram_counts: list[dict[str, int]], output_file: str):
    with open(output_file, "w") as f:
        for count_dict in ngram_counts:
            entries = []
            for key, count in count_dict.items():
                entries.append("|".join(key) + f"={count}")
            f.write("/".join(entries) + "\n")

def load_ngram_counts(file_path: str) -> list[dict[str, int]]:
    with open(file_path, "r") as f:
        raw_dicts = f.readlines()

    result = []
    for raw_dict in raw_dicts:
        clean_dict = {}
        pairs = raw_dict.split("/")
        for pair in pairs:
            raw_key, value = pair.split("=")
            clean_dict[tuple(raw_key.split("|"))] = int(value)
        result.append(clean_dict)
    
    return result
def sample_from_distribution(probs: dict[str, float]) -> str:
    """Given a dict of word:probability, sample a random word using probabilities as weights"""
    words = list(probs.keys())
    weights = list(probs.values())
    return random.choices(words, weights=weights, k=1)[0]

def get_word_probability(word: str, context: list[str], ngram_counts: list[dict[str, int]]) -> float:
    """Calculate the probability of a provided word given context"""
    n = len(context) + 1
    # Try descending "depths" of n until one matches
    for n in range(n, 0, -1):
        ctx = context[-(n - 1):] if n > 1 else []
        key = tuple(ctx + [word])
        if key in ngram_counts[n - 1]:
            numerator = ngram_counts[n - 1][key]
            if n == 1:
                return numerator / sum(ngram_counts[0].values())            
            denominator = sum(
                count for k, count in ngram_counts[n - 1].items() if k[:-1] == tuple(ctx)
            )
            return numerator / denominator

    return 0.0

def choose_next_word(context: list[str], ngram_counts: list[dict[tuple[str], int]]):
    assert len(context) + 1 <= len(ngram_counts) 
    context = [w.upper() for w in context]
    # Calculate all word probabilities given context
    probs = {
        word: get_word_probability(word, context, ngram_counts) for (word,) in ngram_counts[0]
    }
    # Remove any impossible words
    probs = {w: p for w, p in probs.items() if p > 0}
    
    next_word = sample_from_distribution(probs)
    return next_word

def generate_sentence(length: int, ngram_counts: list[dict[tuple[str], int]], seed_str: str = ""):
    seed = seed_str.upper().split()
    max_order = len(ngram_counts)
    result = seed
    for i in range(len(seed), length):
        # Generate the next word
        context = result[max(0, i - (max_order - 1)) : i]
        next_word = choose_next_word(context, ngram_counts)
        result.append(next_word)
    return " ".join(result)

def main():
    BUILD_NGRAMS = False
    if BUILD_NGRAMS:
        words = get_clean_words()
        special = ("<s>", "</s>")
        actual_words = [w for w in words if w not in special]

        # Count all the n-grams
        n=20
        ngram_counts = build_ngram_counts(actual_words, n)
        save_ngram_counts(ngram_counts, f"saved_{n}_gram_counts.txt")
    else:
        ngram_counts = load_ngram_counts("saved_20_gram_counts.txt")

    # Generate some words!
    sentence = generate_sentence(100, ngram_counts, "")
    print(sentence)

if __name__ == "__main__":
    main()