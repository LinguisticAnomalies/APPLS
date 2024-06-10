import random
random.seed(2000)
from nltk.tokenize import sent_tokenize
from tqdm import tqdm
import itertools

class SentencesShuffle4Coherent():
    def __init__(self):
        self.text_str = None
        self.sentences = None
        self.pre_dist = None

    def _calculate_distance(self, a, b):
        return sum([abs(a[i] - b[i]) for i in range(len(a))])

    def _max_distance(self, a):
        b = a.copy()
        b.reverse()
        return self._calculate_distance(a, b)

    def _iter_permutation_list(self):
        iter_list = []
        list_num_sent = list(range(len(self.sentences)))
        count = 0
        for p in itertools.permutations(list_num_sent):
            if count > 10000: # capture outlier
                break
            else:
                count += 1
                iter_list.append(p)
        return iter_list    

    def _get_shuffle_list(self):
        shuffle_list = self._iter_permutation_list()
        shuffle_list_1st = shuffle_list[0]
        # random select 5 from shuffle_list
        if len(shuffle_list) > 10:
            shuffle_list_select = random.sample(shuffle_list[1:], 10)
            # combine shuffle_list_select and shuffle_list_filter
            shuffle_list_filter = [shuffle_list_1st] + shuffle_list_select
        else:
            shuffle_list_filter = shuffle_list
        return shuffle_list_filter

    def _reorder_text(self, iter_list):
        return [' '.join([self.sentences[i] for i in p]) for p in iter_list]

    def perturb_iteration(self, text):
        self.text_str = text
        self.sentences = sent_tokenize(self.text_str)
        if len(self.sentences) <= 1:
            return self.text_str, 0, None
        iter_list = self._get_shuffle_list()
        dist_list = [self._calculate_distance(list(range(len(self.sentences))), p) for p in iter_list]
        dist_percent_list = [dist_list[i]/self._max_distance(list(range(len(self.sentences)))) for i in range(len(dist_list))]
        perturb_text_list = [self._reorder_text(iter_list)[i] for i in range(len(dist_list))]
        return perturb_text_list, dist_percent_list, iter_list