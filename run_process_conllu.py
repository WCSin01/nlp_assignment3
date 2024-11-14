from functions import pickle_dump
from process_conllu import ConlluProcessor

if __name__ == "__main__":
    dataset = ConlluProcessor.process_conllu_for_hmm("ptb-train.conllu")
    pickle_dump(dataset, "checkpoints/dataset.pkl")

    sentences = ConlluProcessor.process_conllu_for_bert("ptb-train.conllu")
    pickle_dump(sentences, "checkpoints/sentences.pkl")