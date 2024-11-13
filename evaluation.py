# v_measure:
# a permutation of the class or cluster label values won’t change the score value in any way.
# symmetric: switching label_true with label_pred will return the same score value.

from sklearn.metrics.cluster import v_measure_score
from utils import calculate_variation_of_information

