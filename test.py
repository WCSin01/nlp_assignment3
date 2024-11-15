from functions import pickle_dump, pickle_load

n_sentences = 39650

if __name__ == "__main__":
    tokens = []
    for i in range(n_sentences):
        sentence_tokens: list[str] = pickle_load(f"checkpoints/bert_tokens/sentence{i}.pkl")
        tokens.append(sentence_tokens)

        if i % 5000 == 0:
            print(i)
    pickle_dump(tokens, "tokens.pkl")