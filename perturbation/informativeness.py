import random
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer
from nltk.cluster.util import cosine_distance
from keybert import KeyBERT
import re
import json
random.seed(2000)
MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+", re.UNICODE)
def normalize_whitespace(text):
    """
    Translates multiple whitespace into single space character.
    If there is at least one new line character chunk is replaced
    by single LF (Unix new line) character.
    """
    return MULTIPLE_WHITESPACE_PATTERN.sub(_replace_whitespace, text)


def _replace_whitespace(match):
    text = match.group()
    if "\n" in text or "\r" in text:
        return "\n"
    else:
        return " "


def is_blank(string):
    """
    Returns `True` if string contains only white-space characters
    or is empty. Otherwise `False` is returned.
    """
    return not string or string.isspace()


def get_symmetric_matrix(matrix):
    """
    Get Symmetric matrix
    :param matrix:
    :return: matrix
    """
    return matrix + matrix.T - np.diag(matrix.diagonal())


def core_cosine_similarity(vector1, vector2):
    """
    measure cosine similarity between two vectors
    :param vector1:
    :param vector2:
    :return: 0 < cosine similarity value < 1
    """
    return 1 - cosine_distance(vector1, vector2)

class TextRank4Sentences():
    def __init__(self):
        self.damping = 0.85  # damping coefficient, usually is .85
        self.min_diff = 1e-5  # convergence threshold
        self.steps = 100  # iteration steps
        self.text_str = None
        self.sentences = None
        self.pr_vector = None

    def _sentence_similarity(self, sent1, sent2, stopwords=None):
        if stopwords is None:
            stopwords = []

        sent1 = [w.lower() for w in sent1]
        sent2 = [w.lower() for w in sent2]

        all_words = list(set(sent1 + sent2))

        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        # build the vector for the first sentence
        for w in sent1:
            if w in stopwords:
                continue
            vector1[all_words.index(w)] += 1

        # build the vector for the second sentence
        for w in sent2:
            if w in stopwords:
                continue
            vector2[all_words.index(w)] += 1

        return core_cosine_similarity(vector1, vector2)

    def _build_similarity_matrix(self, sentences, stopwords=None):
        # create an empty similarity matrix
        sm = np.zeros([len(sentences), len(sentences)])

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2:
                    continue

                sm[idx1][idx2] = self._sentence_similarity(sentences[idx1], sentences[idx2], stopwords=stopwords)

        # Get Symmeric matrix
        sm = get_symmetric_matrix(sm)

        # Normalize matrix by column
        norm = np.sum(sm, axis=0)
        sm_norm = np.divide(sm, norm, where=norm != 0)  # this is to ignore the 0 element in norm

        return sm_norm

    def _run_page_rank(self, similarity_matrix):

        pr_vector = np.array([1] * len(similarity_matrix))

        # Iteration
        previous_pr = 0
        for epoch in range(self.steps):
            pr_vector = (1 - self.damping) + self.damping * np.matmul(similarity_matrix, pr_vector)
            if abs(previous_pr - sum(pr_vector)) < self.min_diff:
                break
            else:
                previous_pr = sum(pr_vector)

        return pr_vector

    def _get_sentence(self, index):

        try:
            return self.sentences[index]
        except IndexError:
            return ""

    def get_top_sentences(self, number=1):

        top_sentences = {}

        if self.pr_vector is not None:

            sorted_pr = np.argsort(self.pr_vector)
            sorted_pr = list(sorted_pr)
            sorted_pr.reverse()

            index = 0
            index_list = []
            for epoch in range(number):
                # print (str(sorted_pr[index]) + " : " + str(self.pr_vector[sorted_pr[index]]))
                sent = self.sentences[sorted_pr[index]]
                sent = normalize_whitespace(sent)
                top_sentences[sent] = self.pr_vector[sorted_pr[index]]
                index_list.append(sorted_pr[index])
                index += 1
        return top_sentences, index_list

    def analyze(self, text, stop_words=None):
        self.text_str = text
        self.sentences = sent_tokenize(self.text_str)

        tokenized_sentences = [word_tokenize(sent) for sent in self.sentences]

        similarity_matrix = self._build_similarity_matrix(tokenized_sentences, stop_words)

        self.pr_vector = self._run_page_rank(similarity_matrix)

class DeleteSentence(TextRank4Sentences):
    def __init__(self):
        super(DeleteSentence, self).__init__()

    def delete_sentence(self, number):
        if number > len(self.sentences):
            number = len(self.sentences)
            top_sentences, index_list = self.get_top_sentences(number=number)
            delete_sent_percent = 1.0
        else:
            top_sentences, index_list = self.get_top_sentences(number=number)
            delete_sent_percent = number / len(self.sentences)
        kept_sentences = [self.sentences[i] for i in range(len(self.sentences)) if i not in index_list]
        delete_word_percent = (len(word_tokenize(self.text_str)) - len(word_tokenize(' '.join(kept_sentences)))) / len(word_tokenize(self.text_str))
        return ' '.join(kept_sentences), delete_sent_percent, delete_word_percent, top_sentences

    def perturb_iteration(self, text):
        self.analyze(text)
        new_claims_list = []
        perturb_sent_percent = []
        perturb_word_percent = []
        replaced_token_list = []
        max_num = len(self.sentences)
        for delete_num in range(max_num):
            new_claim, delete_sent_percent, delete_word_percent, top_sentences = self.delete_sentence(delete_num)
            new_claims_list.append(new_claim)
            perturb_sent_percent.append(delete_sent_percent)
            perturb_word_percent.append(delete_word_percent)
            replaced_token_list.append(top_sentences)
        return new_claims_list, perturb_sent_percent, perturb_word_percent, replaced_token_list

class AddSentence(TextRank4Sentences):
    def __init__(self):
        super(AddSentence, self).__init__()

    def add_non_related_sentence(self, number, external_sentences):
        select_list = random.sample(external_sentences, number)
        # delete \n at the end of sentence
        select_list = [i.replace('\n', '') for i in select_list]
        added_sent_percent = number / len(self.sentences)
        added_word_percent = len(word_tokenize(' '.join(select_list))) / len(word_tokenize(self.text_str))
        # insert sentence from select_list to self.sentences
        edited_sentences = self.sentences.copy()
        for i in range(len(select_list)):
            edited_sentences.insert(random.randint(0, len(edited_sentences)), select_list[i])
        return ' '.join(edited_sentences), added_sent_percent, added_word_percent, select_list

    def perturb_iteration(self, text, external_sentences):
        self.analyze(text)
        new_claims_list = []
        perturb_sent_percent = []
        perturb_word_percent = []
        replaced_token_list = []
        max_num = len(self.sentences)
        for add_num in range(max_num + 1):
            new_claim, add_sent_percent, add_word_percent, top_sentences = self.add_non_related_sentence(add_num, external_sentences)
            new_claims_list.append(new_claim)
            perturb_sent_percent.append(add_sent_percent)
            perturb_word_percent.append(add_word_percent)
            replaced_token_list.append(top_sentences)
        return new_claims_list, perturb_sent_percent, perturb_word_percent, replaced_token_list

class AddDefinition:
    def __init__(self):
        self.text_str = None
        self.keywords = None

    def extract_keywords(self, text, num_keywords):
        """
        Extract keywords from text
        :param text:
        :return: keywords
        """
        self.text_str = text
        keybert = KeyBERT()
        keyword = keybert.extract_keywords(self.text_str, top_n=num_keywords, stop_words='english')
        self.keywords = [i[0] for i in keyword]
        # lower case keywords
        self.keywords = [i.lower() for i in self.keywords]
        
    def add_definitions(self, external_definitions):
        """
        Add definitions from external source
        """
        descriptions = []
        count = 0
        if len(self.keywords) > 0:
            for key in self.keywords:
                try:
                    tmp = external_definitions[key] + ' '
                    descriptions.append(tmp)
                    success = True
                except:
                    descriptions.append('')
                    success = False
                if success:
                    count += 1
            # add descriptions to text
            # print('add descriptions to text')
            entity = self.keywords.copy()
            definition = descriptions.copy()
            abs_wiki_text = ''
            for start, end in PunktSentenceTokenizer().span_tokenize(self.text_str):
                abs_wiki_text += self.text_str[start:end] + ' '
                for e in entity:
                    if e in abs_wiki_text[start:end]:
                            index = entity.index(e)
                            if definition[index] is not None:
                                    abs_wiki_text += definition[index]
                            entity.pop(index)
                            definition.pop(index)
        else:
            abs_wiki_text = self.text_str
        return abs_wiki_text, count, self.keywords

    def perturb_iteration(self, text, external_definitions, top_n=3):
        new_claims_list = []
        perturb_word_num = []
        replaced_token_list = []
        for add_num in range(1, top_n + 1):
            self.extract_keywords(text, add_num)
            new_claim, add_definition_num, keywords = self.add_definitions(external_definitions)
            new_claims_list.append(new_claim)
            perturb_word_num.append(add_definition_num)
            replaced_token_list.append(keywords)
        return new_claims_list, perturb_word_num, replaced_token_list

