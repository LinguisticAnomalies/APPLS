# conda activate claim-generation-test

from transformers import pipeline, FillMaskPipeline, TextClassificationPipeline, PreTrainedTokenizer
from transformers import RobertaTokenizerFast, RobertaForMaskedLM, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer
from nltk.tokenize import sent_tokenize
import pandas as pd
import torch
import numpy as np
from tqdm import tqdm
import spacy
import scispacy
from scispacy.linking import EntityLinker
from collections import defaultdict
import random
random.seed(2000)
from string import punctuation
from scipy.spatial.distance import cosine
from typing import List, AnyStr

scientific_claim_folder = './'

nlp = spacy.load('en_core_sci_md')
nlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
linker = nlp.get_pipe("scispacy_linker")
tui_to_cui = defaultdict(list)
for cui in linker.kb.cui_to_entity:
    for t in linker.kb.cui_to_entity[cui].types:
        tui_to_cui[t].append(cui)

cui_to_rel = {}
rel_to_cui = defaultdict(set)
with open(scientific_claim_folder + 'scientific-claim-generation/MRCONSO.RRF') as f:
    for l in f:
        fields = l.strip().split('|')
        cui_to_rel[fields[0]] = fields[12]
        rel_to_cui[fields[12]].add(fields[0])

cui2vec = {}
with open(scientific_claim_folder + 'scientific-claim-generation/cui2vec_pretrained.csv') as f:
    next(f)
    for l in f:
        fields = l.strip().split(',')
        cui2vec[fields[0][1:-1]] = np.array(fields[1:], dtype=np.float32)

nli = pipeline('sentiment-analysis', model='roberta-large-mnli', return_all_scores=True, device=0, max_length=512)
lm = AutoModelForCausalLM.from_pretrained('gpt2')
lm_tk = AutoTokenizer.from_pretrained('gpt2')

device = torch.device("cpu")
if torch.cuda.is_available():
    print("Training on GPU")
    device = torch.device("cuda:0")

class EntitySwap:
    def get_perplexity(self, 
            sentences: List[AnyStr],
            lm: GPT2LMHeadModel,
            lm_tokenizer: PreTrainedTokenizer,
            device: torch.device) -> List[float]:
        """
        Get the perplexity for a set of sentences using a given language model
        :param sentences:
        :param lm:
        :param lm_tokenizer:
        :param device:
        :return:
        """
        lm.eval()
        lm.to(device)
        ppl = []
        for sent in sentences:
            inputs = lm_tokenizer.batch_encode_plus([sent])
            with torch.no_grad():
                loss = lm(torch.tensor(inputs['input_ids']).to(device), labels=torch.tensor(inputs['input_ids']).to(device))[0]
                ppl.append(np.exp(loss.cpu().item()))
        lm.to('cpu')
        return ppl


    def kbin( self,
            claims: List[str],
            nli: TextClassificationPipeline,
            language_model: GPT2LMHeadModel,
            lm_tokenizer: PreTrainedTokenizer,
            device: torch.device,
            n_samples: int = 3) -> List[List]:
        """
        Generate negative claims by extracting named entities, linking them to UMLS, gathering all related concepts to the entity, ranking the
        concepts by measuring their cosine distance from the concept for the extracted named entity using cui2vec vectors, selecting
        the text form of those concepts to use by replacing them in the original text and ranking them based on perplexity,
        then sampling from the top concepts and selecting the replacement which has the strongest contradiction with the original claim
        using an external NLI model
        :param claims:
        :param nli:
        :param language_model:
        :param lm_tokenizer:
        :param device:
        :param n_samples:
        :return:
        """
        data = []
        for claim in claims:
            # print('claim: ', claim)
            suggs = []
            curr_claims = []
            # print('entities', nlp(claim).ents)
            for ent in nlp(claim).ents: 
                if len(ent._.kb_ents) > 0:
                    cui = ent._.kb_ents[0][0]
                    if cui not in cui2vec:
                        continue
                    tui = linker.kb.cui_to_entity[cui].types[0]
                    alias_options = []
                    cui_options = list(set(tui_to_cui[tui]) - set([cui]))
                    if cui in cui_to_rel:
                        cui_options = list(set(cui_options) & rel_to_cui[cui_to_rel[cui]])
                        cui_options = list(set(cui_options) & set(cui2vec.keys()))
                    # Calculate distance
                    try:
                        dist = [cosine(cui2vec[cui], cui2vec[opt]) for opt in cui_options]
                        cui_options = [cui_options[idx] for idx in np.argsort(dist)]
                    except:
                        continue

                    j = 0
                    while len(alias_options) < n_samples and j < len(cui_options):
                        aliases_curr = [alias.lower() for alias in linker.kb.cui_to_entity[cui_options[j]].aliases + [linker.kb.cui_to_entity[cui_options[j]].canonical_name] if len(alias) < (len(ent.text) * 2) and not any(p in alias for p in punctuation)]

                        if len(aliases_curr) > 0:
                            # Rank by perplexity
                            sents_curr = [[claim.replace(ent.text, alias), (ent.text, alias)] for alias in aliases_curr]
                            ppl = self.get_perplexity([s[0] for s in sents_curr], language_model, lm_tokenizer, device)
                            alias_options.append(sents_curr[np.argmin(ppl)])
                        j += 1
                    suggs.extend(alias_options)
            # print('suggs', suggs)
            for sug in suggs:
                if sug[0] != claim.lower():
                    if len(sug[0].split()) > 512:
                    # keep the first 512 tokens of sug[0]
                        sug[0] = ' '.join(sug[0].split()[:512])
                    score = nli(f"{claim}</s></s>{sug[0]}")[0][0]['score']
                    curr_claims.append([claim, sug[1], sug[0], score])
            if len(curr_claims) > 0:
                top_claim = list(sorted(curr_claims, key=lambda x: x[-1], reverse=True))[0]
            else:
                top_claim = None
            # print('top_claim', top_claim)
            data.append(top_claim)
            # print('data', data)

        return data

    def get_candidate_sents(self, text):
        original_sents = sent_tokenize(text)
        candidate_sents = self.kbin(original_sents, nli, lm, lm_tk, device, n_samples = len(original_sents)+1)
        return original_sents, candidate_sents

    def perturb_iteration(self, item):
        original_sents, candidate_sents = self.get_candidate_sents(item)
        if not candidate_sents:
            return item, 0.0, None
        new_claims_list = []
        perturb_word_num = []
        replaced_token_list = []
        new_claims_list.append(item)
        perturb_word_num.append(0.0)
        replaced_token_list.append(None)
        candidate_sents = [c for c in candidate_sents if c is not None]
        original_sents_with_entities_index = [i for i, s in enumerate(candidate_sents) if s is not None]
        for num in range(1, len(candidate_sents) + 1): # replace sentence in original_sents_with_entities_index
            output_text = original_sents
            selected_sentence_index = np.random.choice(original_sents_with_entities_index, num, replace=False)
            selected_sentence_index = np.sort(selected_sentence_index)
            # print('selected_sentence_index', selected_sentence_index)
            # print('candidate_sents', candidate_sents)
            # print('candidate_sents[selected_sentence_index]', candidate_sents[selected_sentence_index[0]])
            entities_swapped = [candidate_sents[i][1] for i in selected_sentence_index]
            replaced_text = [candidate_sents[i][2] for i in selected_sentence_index]
            for i in range(len(selected_sentence_index)):
                output_text[selected_sentence_index[i]] = replaced_text[i]
            new_claims_list.append(' '.join(output_text))
            perturb_word_num.append(num/len(original_sents))
            replaced_token_list.append(entities_swapped)
        return new_claims_list, perturb_word_num, replaced_token_list