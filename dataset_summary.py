import math
import numpy as np
from collections import Counter
from process_conllu import ConlluProcessor
import matplotlib.pyplot as plt


word_count_dict, sentence_lengths = ConlluProcessor.process_conllu_for_inspection("ptb-train.conllu")
word_counts = sorted([word_count_dict[word] for word in word_count_dict], reverse=True)
n_words_with_count = Counter(word_counts)
proportion_words_with_count = np.zeros(len(n_words_with_count))
for i, (count, n_words) in enumerate(n_words_with_count.items()):
    proportion_words_with_count[i] = count * n_words
proportion_words_with_count = proportion_words_with_count / np.sum(proportion_words_with_count)
cumulative_proportion = np.cumsum(proportion_words_with_count[::-1])
for count, proportion in zip(list(n_words_with_count.keys())[::-1], cumulative_proportion):
    print((count, proportion.item()))

fig, ax = plt.subplots()
# s = marker size
ax.scatter(np.arange(1, len(word_counts)+1), word_counts, s=2)
ax.set_yscale("log")
ax.set_title("Zipf's law")
ax.set_ylabel("Word frequency")
ax.set_xlabel("Rank")
plt.savefig("zipf.png")
plt.show()

# 19000/949701 = 0.02
# 5000/949701 * 2 = 0.01
#