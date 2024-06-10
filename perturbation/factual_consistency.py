from nltk.tokenize import word_tokenize, sent_tokenize
import random
# random.seed(1000)
random.seed(2000)
import spacy
import nltk
import pattern
from pattern.en import tenses
import collections

from transformers import pipeline, FillMaskPipeline, TextClassificationPipeline, PreTrainedTokenizer
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
from tqdm import tqdm
import scispacy
from scispacy.linking import EntityLinker
from collections import defaultdict

from string import punctuation
from scipy.spatial.distance import cosine
from typing import List, AnyStr
from nltk.corpus import wordnet

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

nlp = spacy.load('en_core_web_sm')
nltk.data.path.append("/home/NETID/yguo50/nltk_data_new") # trick to fix my server issue

def shuffle_list_without_replacement(input_list):
    # shuffle list without replacement
    output_list = []
    copy_input_list = input_list.copy()
    while len(copy_input_list) > 0:
        random_idx = random.randint(0, len(copy_input_list) - 1)
        output_list.append(copy_input_list[random_idx])
        copy_input_list.pop(random_idx)
    # check whether the output list is the same as the input list
    for i in range(len(input_list)):
        if input_list[i] == output_list[i]:
            # if the same, shuffle again
            return shuffle_list_without_replacement(input_list)
    return output_list

def align_ws(old_token, new_token):
    output = []
    # Align trailing whitespaces between tokens
    for i in range(len(old_token)):
        if old_token[i][-1] == new_token[i][-1] == " ":
            output.append(new_token[i])
        elif old_token[i][-1] == " ":
            output.append(new_token[i] + " ")
        elif new_token[i][-1] == " ":
            output.append(new_token[i][:-1])
        else:
            output.append(new_token[i])
    return output

def align_tense(old_token, new_token):
    # align tense of new_token with old_token  
    output_token = []
    for i in range(len(old_token)):
        new_token_text = nlp(new_token[i].split(' ')[0])
        # -ing
        if old_token[i].text.endswith('ing'):
            if new_token_text.text.endswith('ing'):
                output_token.append(new_token[i])
            else:
                new_token_present = pattern.en.conjugate(new_token_text.text, tense='present')
                if new_token_present.endswith('e'):
                    output_token.append(new_token_present[:-1] + 'ing')
                else:
                    output_token.append(new_token_present + 'ing')
        else:
            try:
                old_tenses = pattern.en.lexeme(old_token[i].text)
                tense_idx = old_tenses.index(old_token[i].text)
                new_tenses = pattern.en.lexeme(new_token_text.text)
                new_tense = new_tenses[tense_idx]
            except: # if new_tenses does not have the same tense as old_tenses
                new_tense = pattern.en.lexeme(pattern.en.conjugate(new_token_text.text, tense='infinitive'))[0]
            output_token.append(new_tense)
    return output_token

class NumberSwap:
    # NER swapping class specialized for numbers (excluding dates)
    def __init__(self):
        self.categories = ("PERCENT", "MONEY", "QUANTITY", "CARDINAL")

    def get_entities(self, text):
        # find entities in given category
        claim_tokens = [token for token in text if token.ent_type_ in self.categories]
        # keep only entities that are digit
        claim_tokens_filtered = [ent for ent in claim_tokens if ent.text.isdigit()]
        return claim_tokens_filtered

    def swap_entities(self, num_swap, text):
        claim_tokens_filtered = self.get_entities(text)
        if not claim_tokens_filtered:
            return text.text, None, None
        if len(claim_tokens_filtered) < num_swap:
            return text.text, None, None
        if num_swap == 0:
            return text.text, None, None

        sample_idx = random.sample(range(len(claim_tokens_filtered)), num_swap)
        replaced_token = [claim_tokens_filtered[i] for i in sample_idx]
        replaced_token_text_with_ws = [claim_tokens_filtered[i].text_with_ws for i in sample_idx]

        swapped_token = [str(int(i.text) + random.randint(1, 5)) for i in replaced_token]

        # swapped_token = [str(int(i.text) + 5) for i in replaced_token]
        swapped_token_align_ws = align_ws(replaced_token_text_with_ws, swapped_token)

        text_tokens = [token.text_with_ws for token in text]
        token_idx_list = []
        for token in range(len(replaced_token)):
            token_idx = replaced_token[token].i
            text_tokens[token_idx] = str(swapped_token_align_ws[token])
            token_idx_list.append(token_idx)

        new_claim = "".join(text_tokens)
        return new_claim, replaced_token, token_idx_list

    def perturb_iteration(self, text):
        new_claims = []
        perturb_percent = []
        replaced_token_list = []
        replaced_token_idx_list = []
        claim_tokens_filtered = self.get_entities(text)
        
        if not claim_tokens_filtered:
            new_claims.append(text.text)
            perturb_percent.append(0)
            return new_claims, perturb_percent, None, None
        for i in range(len(claim_tokens_filtered) + 1):
            try:
                perturbed_claim, replaced_token, token_idx = self.swap_entities(i, text)
                new_claims.append(perturbed_claim)
                perturb_percent.append(i / len(claim_tokens_filtered))
                replaced_token_list.append(replaced_token)
                replaced_token_idx_list.append(token_idx)
            except:
                continue
        # print("claim_tokens_filtered: ", claim_tokens_filtered)
        # print("NumberSwap: ", len(new_claims))
        # print('perturb_percent: ', perturb_percent)
        # print('replaced_token_list: ', replaced_token_list)
        return new_claims, perturb_percent, replaced_token_list, replaced_token_idx_list

class VerbSwap:
    def __init__(self):
        self.categories = ("VERB")

    def get_entities(self, text):
        claim_tokens = [ent for ent in text if ent.pos_ in self.categories]
        claim_tokens_text = [x.text for x in claim_tokens]
        # find index of duplicate claim_tokens_text and remove those index from claim_tokens
        duplicate_idx = [i for i, x in enumerate(claim_tokens_text) if claim_tokens_text.count(x) > 1]
        claim_tokens = [claim_tokens[i] for i in range(len(claim_tokens)) if i not in duplicate_idx]
        return claim_tokens

    def swap_entities(self, num_swap, text):
        claim_tokens = self.get_entities(text)
        # claim_tokens = [x for x in text if x.dep_ in ['csubj', 'nsubj']]
        if not claim_tokens:
            return text.text, None, None
        if num_swap == 0:
            return text.text, None, None
        if len(claim_tokens) < num_swap:
            return text.text, None, None

        if num_swap == 1:
            sample_idx = random.sample(range(len(claim_tokens)), num_swap + 1)
            replaced_token = [claim_tokens[i] for i in sample_idx]
            replaced_token_text_with_ws = [claim_tokens[i].text_with_ws for i in sample_idx]
            swapped_token = shuffle_list_without_replacement(replaced_token_text_with_ws)
            try:
                swapped_token_align_tense = align_tense(replaced_token, swapped_token)
                swapped_token_align_ws = align_ws(replaced_token_text_with_ws, swapped_token_align_tense)
            except:
                swapped_token_align_ws = swapped_token

            replaced_token = replaced_token[:1]
            swapped_token_align_ws = swapped_token_align_ws[:1]
        else:
            sample_idx = random.sample(range(len(claim_tokens)), num_swap)
            replaced_token = [claim_tokens[i] for i in sample_idx]
            replaced_token_text_with_ws = [claim_tokens[i].text_with_ws for i in sample_idx]
            
            swapped_token = shuffle_list_without_replacement(replaced_token_text_with_ws)
            try:
                swapped_token_align_tense = align_tense(replaced_token, swapped_token)
                swapped_token_align_ws = align_ws(replaced_token_text_with_ws, swapped_token_align_tense)
            except:
                swapped_token_align_ws = swapped_token

        text_tokens = [token.text_with_ws for token in text]
        token_idx_list = []
        for token in range(len(replaced_token)):
            token_idx = replaced_token[token].i
            text_tokens[token_idx] = str(swapped_token_align_ws[token])
            token_idx_list.append(token_idx)

        new_claim = "".join(text_tokens)
        return new_claim, replaced_token, token_idx_list

    def perturb_iteration(self, text):
        new_claims = []
        perturb_percent = []
        replaced_token_list = []
        replaced_token_idx_list = []
        claim_tokens_filtered = self.get_entities(text)
        if not claim_tokens_filtered:
            new_claims.append(text.text)
            perturb_percent.append(0)
            return new_claims, perturb_percent, None, None
        if len(claim_tokens_filtered) == 1:
            new_claims.append(text.text)
            perturb_percent.append(0)
            return new_claims, perturb_percent, None, None
        for i in range(len(claim_tokens_filtered) + 1):
            perturbed_claim, replaced_token, token_idx = self.swap_entities(i, text)
            new_claims.append(perturbed_claim)
            perturb_percent.append(i / len(claim_tokens_filtered))
            replaced_token_list.append(replaced_token)
            replaced_token_idx_list.append(token_idx)
        return new_claims, perturb_percent, replaced_token_list, replaced_token_idx_list

class NegateSentences:
    # Apply or remove negation from negatable tokens
    def get_sent_num(self, text):
        sent_len_text = sum(1 for _ in text.sents)
        return sent_len_text

    def add_negation(self, num_swap, text):
        sent_len_text = self.get_sent_num(text)
        if num_swap == 0:
            return text.text, None, None
        if sent_len_text < num_swap:
            return text.text, None, None

        sample_index = random.sample(range(sent_len_text), num_swap)
        text_tokens = [token.text_with_ws for token in text]
        selected_sent_num = 0

        for sentence in text.sents:
            if selected_sent_num not in sample_index:
                selected_sent_num += 1
                continue
            else:
                selected_sent_num += 1 
                root_id = [x.i for x in sentence if x.dep_ == 'ROOT'][0]
                root = text[root_id]

                if '?' in sentence.text and sentence[0].text.lower() == 'how':
                    continue
                if root.lemma_.lower() in ['thank', 'use']:
                    continue
                if root.pos_ not in ['VERB', 'AUX']:
                    continue
                neg = [True for x in sentence if x.dep_ == 'neg' and x.head.i == root_id]
                if neg:
                    continue
                if root.lemma_ == 'be':
                    if '?' in sentence.text:
                        continue
                    if root.text.lower() in ['is', 'was', 'were', 'am', 'are', '\'s', '\'re', '\'m']:
                        # append not after root.text
                        text_tokens[root_id] = text_tokens[root_id] + 'not '
                    else:
                        text_tokens[root_id] = 'not ' + text_tokens[root_id]
                else:
                    aux = [x for x in sentence if x.dep_ in ['aux', 'auxpass'] and x.head.i == root_id]
                    if aux:
                        aux = aux[0]
                        if aux.lemma_.lower() in ['can', 'do', 'could', 'would', 'will', 'have', 'should']:
                            lemma = text[aux.i].lemma_.lower()
                            if lemma == 'will':
                                fixed = 'won\'t'
                            elif lemma == 'have' and text[aux.i].text in ['\'ve', '\'d']:
                                fixed = 'haven\'t' if text[aux.i].text == '\'ve' else 'hadn\'t'
                            elif lemma == 'would' and text[aux.i].text in ['\'d']:
                                fixed = 'wouldn\'t'
                            else:
                                fixed = text[aux.i].text.rstrip('n') + 'n\'t' if lemma != 'will' else 'won\'t'
                            fixed = '%s ' % fixed
                            text_tokens[aux.i] = fixed
                        text_tokens[root_id] = 'not ' + text_tokens[root_id]
                    else:
                        subj = [x for x in sentence if x.dep_ in ['csubj', 'nsubj']]
                        try:
                            p = pattern.en.tenses(root.text)
                            tenses = collections.Counter([x[0] for x in pattern.en.tenses(root.text)]).most_common(1)
                            tense = tenses[0][0] if len(tenses) else 'present'
                            params = [tense, 3]
                            if p:
                                tmp = [x for x in p if x[0] == tense]
                                if tmp:
                                    params = list(tmp[0])
                                else:
                                    params = list(p[0])
                            if root.tag_ not in ['VBG']:
                                do = pattern.en.conjugate('do', *params) + 'n\'t'
                                new_root = pattern.en.conjugate(root.text, tense='infinitive')
                            else:
                                do = 'not'
                                new_root = root.text
                            text_tokens[root_id] = '%s %s ' % (do, new_root)
                        except:
                            # print('pattern.en.tenses error', root.text, text)
                            text_tokens[root_id] = 'not ' + text_tokens[root_id]
        return ''.join(text_tokens), sample_index, sample_index

    def perturb_iteration(self, text):
        new_claims = []
        perturb_percent = []
        replaced_token_list = []
        replaced_token_idx_list = []
        sent_num = self.get_sent_num(text)
        if not sent_num:
            new_claims.append(text.text)
            perturb_percent.append(0)
            return new_claims, perturb_percent, None, None
        for i in range(sent_num + 1):
            try:
                perturbed_claim, sample_index, sample_index = self.add_negation(i, text)
                new_claims.append(perturbed_claim)
                perturb_percent.append(i / sent_num)
                replaced_token_list.append(sample_index)
                replaced_token_idx_list.append(sample_index)
            except:
                continue
        return new_claims, perturb_percent, replaced_token_list, replaced_token_idx_list

class SynonymsVerbSwap:
    def synonyms_pairs(self, text):
        verb_list = []
        synonyms_list = []
        original_sents = sent_tokenize(text)
        for sent in original_sents:
            tokens = word_tokenize(sent)
            pos_tags = nltk.pos_tag(tokens)
            for i, tag in enumerate(pos_tags):
                if tag[1] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
                    if tag[0] not in ["is", "are", "am"]:
                        verb_list.append(tag[0])
        present_tense_list = []
        for verb in verb_list:
            try:
                present = pattern.en.lexeme(verb)[0]
                present_tense_list.append(present)
            except:
                continue
        if not present_tense_list:
            return None, None
        else:
            present_tense_set = list(set(present_tense_list))
            present_tense_set_idx = [present_tense_list.index(i) for i in present_tense_set]
            replaced_token = [verb_list[i] for i in present_tense_set_idx]
            replaced_token_output = replaced_token.copy()
            for token_index in range(len(replaced_token)):
                i = replaced_token[token_index]
                present_i = present_tense_set[token_index]
                synonyms = wordnet.synsets(i, pos=wordnet.VERB)
                nlp_synonyms = [nlp(syn.lemmas()[0].name()) for syn in synonyms]
                synonyms_present_tense = [pattern.en.lexeme(i.text)[0] for i in nlp_synonyms]
                synonyms_set = list(set(synonyms_present_tense))
                if present_i in synonyms_set:
                    synonyms_set.remove(present_i)
                if len(synonyms_set) == 0:
                    replaced_token_output.remove(i)
                else:
                    synonyms_list.append(random.choice(synonyms_set))
            return replaced_token_output, synonyms_list

    def swap_entities(self, num_swap, text):
        replaced_token_output = []
        replaced_token, swapped_token = self.synonyms_pairs(text)
        if not replaced_token:
            return text, None, None
        if num_swap == 0:
            return text, None, None
        if len(replaced_token) < num_swap:
            return text, None, None
        sample_idx = random.sample(range(len(replaced_token)), num_swap)
        replaced_token = [replaced_token[i] for i in sample_idx]
        swapped_token = [swapped_token[i] for i in sample_idx]
        replaced_token = [nlp(i) for i in replaced_token]
        swapped_token_align_tense = align_tense(replaced_token, swapped_token)

        text_tokens = [token.text for token in nlp(text)]
        text_token_with_space = [token.text_with_ws for token in nlp(text)]
        token_idx_list = []
        for token in range(len(replaced_token)):
            for i in range(len(text_tokens)):
                if str(text_tokens[i]) == str(replaced_token[token]):
                    text_token_with_space[i] = str(swapped_token_align_tense[token]) + ' '
                    replaced_token_output.append(replaced_token[token])
                    token_idx_list.append(i)
        new_claim = "".join(text_token_with_space)
        return new_claim, replaced_token_output, token_idx_list

    def perturb_iteration(self, text):
        new_claims = []
        perturb_percent = []
        replaced_token_list = []
        replaced_token_idx_list = []
        claim_tokens_filtered = self.synonyms_pairs(text)
        if not claim_tokens_filtered[0]:
            new_claims.append(text)
            perturb_percent.append(0)
            return new_claims, perturb_percent, None, None
        for i in range(len(claim_tokens_filtered[0]) + 1):
            perturbed_claim, replaced_token, token_idx = self.swap_entities(i, text)
            new_claims.append(perturbed_claim)
            perturb_percent.append(i / len(claim_tokens_filtered[0]))
            replaced_token_list.append(replaced_token)
            replaced_token_idx_list.append(token_idx)
        return new_claims, perturb_percent, replaced_token_list, replaced_token_idx_list

class AntonymsVerbSwap:
    def synonyms_pairs(self, text):
        verb_list = []
        antonyms_list = []
        original_sents = sent_tokenize(text)
        for sent in original_sents:
            tokens = word_tokenize(sent)
            pos_tags = nltk.pos_tag(tokens)
            for i, tag in enumerate(pos_tags):
                if tag[1] in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:
                    if tag[0] not in ["is", "are", "am"]:
                        verb_list.append(tag[0])
        replaced_token = verb_list.copy()
        for token_index in range(len(verb_list)):
            i = verb_list[token_index]
            synonyms = wordnet.synsets(i, pos=wordnet.VERB)
            antonyms = []
            for syn in synonyms:
                for l in syn.lemmas():
                    if l.antonyms():
                        antonyms.append(l.antonyms()[0].name())
            if len(antonyms) == 0:
                replaced_token.remove(i)
            else:
                # nlp_antonyms = [nlp(ant) for ant in antonyms]
                antonyms_list.append(random.choice(antonyms))
        return replaced_token, antonyms_list

    def swap_entities(self, num_swap, text):
        replaced_token_output = []
        replaced_token, swapped_token = self.synonyms_pairs(text)
        if len(swapped_token) == 0:
            return text, None, None
        if num_swap == 0:
            return text, None, None
        if len(replaced_token) < num_swap:
            return text, None, None
        sample_idx = random.sample(range(len(replaced_token)), num_swap)
        replaced_token = [replaced_token[i] for i in sample_idx]
        swapped_token = [swapped_token[i] for i in sample_idx]
        replaced_token = [nlp(i) for i in replaced_token]
        try:
            swapped_token_align_tense = align_tense(replaced_token, swapped_token)
        except:
            swapped_token_align_tense = swapped_token

        text_tokens = [token.text for token in nlp(text)]
        text_token_with_space = [token.text_with_ws for token in nlp(text)]
        token_idx_list = []
        for token in range(len(replaced_token)):
            for i in range(len(text_tokens)):
                if str(text_tokens[i]) == str(replaced_token[token]):
                    text_token_with_space[i] = str(swapped_token_align_tense[token]) + ' '
                    replaced_token_output.append(replaced_token[token])
                    token_idx_list.append(i)
        new_claim = "".join(text_token_with_space)
        return new_claim, replaced_token_output, token_idx_list

    def perturb_iteration(self, text):
        new_claims = []
        perturb_percent = []
        replaced_token_list = []
        replaced_token_idx_list = []
        _, antonyms_list = self.synonyms_pairs(text)
        if len(antonyms_list) == 0:
            new_claims.append(text)
            perturb_percent.append(0)
            return new_claims, perturb_percent, None, None
        else:
            new_claims.append(text)
            perturb_percent.append(0)
            replaced_token_list.append(None)
            replaced_token_idx_list.append(None)
        for i in range(1, len(antonyms_list) + 1):
            perturbed_claim, replaced_token, token_idx = self.swap_entities(i, text)
            new_claims.append(perturbed_claim)
            perturb_percent.append(i / len(antonyms_list))
            replaced_token_list.append(replaced_token)
            replaced_token_idx_list.append(token_idx)
        return new_claims, perturb_percent, replaced_token_list, replaced_token_idx_list