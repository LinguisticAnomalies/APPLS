import os
import json
import pandas as pd
from perturbation.factual_consistency import NumberSwap, VerbSwap, NegateSentences, SynonymsVerbSwap, AntonymsVerbSwap
from perturbation.coherent import SentencesShuffle4Coherent
from perturbation.simplification import LexicalSimplification
from perturbation.informativeness import DeleteSentence, AddSentence, AddDefinition
import argparse
import spacy
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='', help='path to the input file')
    parser.add_argument('--input_file', type=str, default='plaba_en_de_en.csv', help='input file name')
    parser.add_argument('--output_path', type=str, default='', help='path to the output file')
    parser.add_argument('--task', type=str, default='number_swap', help='number_swap, verb_swap, negate_sentence, coherent')
    parser.add_argument('--mode', type=str, default='perturb', help='perturb or evaluate')
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_path + args.input_file)
    if args.mode == 'evaluate':
        # random select 50 samples
        df = df.sample(n=50, random_state=1)
    batch = df['reference_text'].tolist()
    ids = df['id'].tolist()

    task_list = {   
                "number_swap": NumberSwap(),
                "verb_swap": VerbSwap(),
                "synonyms_verb_swap": SynonymsVerbSwap(),
                "antonyms_verb_swap": AntonymsVerbSwap(),
                "negate_sentence": NegateSentences(),
                "coherent": SentencesShuffle4Coherent(),
                "simplification": LexicalSimplification(),
                "delete_sentence": DeleteSentence(),
                "add_non_related_sentence": AddSentence(),
                "add_related_sentence": AddSentence(),
                "add_definition": AddDefinition()
            }

    template = task_list[args.task]
    output_ids_list = []
    reference_text_list = []
    perturbed_text_list = []
    perturbed_tokens_list = []
    perturbed_tokens_idx_list = []

    print('Perturbing {}...'.format(args.task))

    if args.task in ['number_swap', 'verb_swap', 'negate_sentence']:
        nlp = spacy.load('en_core_web_sm')
        batch = [nlp(text) for text in batch]

    if args.task in ['number_swap', 'verb_swap', 'synonyms_verb_swap', 'antonyms_verb_swap', 'negate_sentence']:
        perturbed_percentage_list = []
        for i in tqdm(range(len(batch))):
            perturbed_text, perturbed_percentage, perturbed_tokens, perturbed_idx = template.perturb_iteration(batch[i])
            if perturbed_tokens:
                for token_idx in range(len(perturbed_tokens)):
                    output_ids_list.append(ids[i])
                    reference_text_list.append(batch[i])
                    perturbed_text_list.append(perturbed_text[token_idx])
                    perturbed_percentage_list.append(perturbed_percentage[token_idx])
                    perturbed_tokens_list.append(perturbed_tokens[token_idx])
                    perturbed_tokens_idx_list.append(perturbed_idx[token_idx])
            else:
                output_ids_list.append(ids[i])
                reference_text_list.append(batch[i])
                perturbed_text_list.append(perturbed_text[0])
                perturbed_percentage_list.append(0.0)
                perturbed_tokens_list.append(perturbed_tokens)
                perturbed_tokens_idx_list.append(perturbed_idx)
            assert len(perturbed_text_list) == len(perturbed_percentage_list) == len(perturbed_tokens_list) == len(output_ids_list), 'perturbed_text_list, perturbed_percentage_list, perturbed_tokens_list, output_ids_list should have the same length'
        df_output = pd.DataFrame({'id': output_ids_list, 'reference_text': reference_text_list, 'perturbed_text': perturbed_text_list, 'perturbed_percentage': perturbed_percentage_list, 'perturbed_tokens': perturbed_tokens_list, 'perturbed_tokens_idx': perturbed_tokens_idx_list})
    elif args.task in ['coherent']:
        perturbed_percentage_list = []
        for i in tqdm(range(len(batch))):
            perturbed_text, perturbed_percentage, perturbed_tokens = template.perturb_iteration(batch[i])
            if perturbed_tokens:
                for token_idx in range(len(perturbed_tokens)):
                    output_ids_list.append(ids[i])
                    reference_text_list.append(batch[i])
                    perturbed_text_list.append(perturbed_text[token_idx])
                    perturbed_percentage_list.append(perturbed_percentage[token_idx])
                    perturbed_tokens_list.append(perturbed_tokens[token_idx])
            else:
                output_ids_list.append(ids[i])
                reference_text_list.append(batch[i])
                perturbed_text_list.append(perturbed_text[0])
                perturbed_percentage_list.append(0.0)
                perturbed_tokens_list.append(perturbed_tokens)
            assert len(perturbed_text_list) == len(perturbed_percentage_list) == len(perturbed_tokens_list) == len(output_ids_list), 'perturbed_text_list, perturbed_percentage_list, perturbed_tokens_list, output_ids_list should have the same length'
        df_output = pd.DataFrame({'id': output_ids_list, 'reference_text': reference_text_list, 'perturbed_text': perturbed_text_list, 'perturbed_percentage': perturbed_percentage_list, 'perturbed_tokens': perturbed_tokens_list})    
    elif args.task == 'simplification':
        batch_simple = df['simple_text'].tolist()
        perturbed_chunk_percentage_list = []
        perturbed_sentence_percentage_list = []
        perturbed_word_percentage_list = []
        for i in tqdm(range(len(batch))):
            perturbed_text, perturbed_chunk_percentage, perturbed_sentence_percentage, perturbed_word_percentage, perturbed_tokens = template.perturb_iteration(batch[i], batch_simple[i])
            for token_idx in range(len(perturbed_tokens)):
                output_ids_list.append(ids[i])
                reference_text_list.append(batch[i])
                perturbed_text_list.append(perturbed_text[token_idx])
                perturbed_chunk_percentage_list.append(perturbed_chunk_percentage[token_idx])
                perturbed_sentence_percentage_list.append(perturbed_sentence_percentage[token_idx])
                perturbed_word_percentage_list.append(perturbed_word_percentage[token_idx])
                perturbed_tokens_list.append(perturbed_tokens[token_idx])
        df_output = pd.DataFrame({'id': output_ids_list, 'reference_text': reference_text_list, 'perturbed_text': perturbed_text_list, 'perturbed_chunk_percentage': perturbed_chunk_percentage_list, 'perturbed_sentence_percentage': perturbed_sentence_percentage_list, 'perturbed_word_percentage': perturbed_word_percentage_list, 'perturbed_tokens': perturbed_tokens_list})
    elif args.task in ['delete_sentence']:
        perturbed_sentence_percentage_list = []
        perturbed_word_percentage_list = []
        for i in tqdm(range(len(batch))):
            perturbed_text, perturbed_sentence_percentage, perturbed_word_percentage, perturbed_tokens = template.perturb_iteration(batch[i])
            for token_idx in range(len(perturbed_tokens)):
                output_ids_list.append(ids[i])
                reference_text_list.append(batch[i])
                perturbed_text_list.append(perturbed_text[token_idx])
                perturbed_sentence_percentage_list.append(perturbed_sentence_percentage[token_idx])
                perturbed_word_percentage_list.append(perturbed_word_percentage[token_idx])
                perturbed_tokens_list.append(perturbed_tokens[token_idx])
        df_output = pd.DataFrame({'id': output_ids_list, 'reference_text': reference_text_list, 'perturbed_text': perturbed_text_list, 'perturbed_sentence_percentage': perturbed_sentence_percentage_list, 'perturbed_word_percentage': perturbed_word_percentage_list, 'perturbed_tokens': perturbed_tokens_list})
    elif args.task in ['add_non_related_sentence', 'add_related_sentence']:
        if args.task == 'add_non_related_sentence':
            external_sentences_path = './external_knowledge_source/ACL-ARC/non_related_sentence.txt'
        else:
            external_sentences_path = './external_knowledge_source/Cochrane-abstract-202208-202301/related_sentence.txt'
        with open(external_sentences_path, 'r') as f:
            external_sentences = f.readlines()
        perturbed_sentence_percentage_list = []
        perturbed_word_percentage_list = []
        for i in tqdm(range(len(batch))):
            perturbed_text, perturbed_sentence_percentage, perturbed_word_percentage, perturbed_tokens = template.perturb_iteration(batch[i], external_sentences)
            for token_idx in range(len(perturbed_tokens)):
                output_ids_list.append(ids[i])
                reference_text_list.append(batch[i])
                perturbed_text_list.append(perturbed_text[token_idx])
                perturbed_sentence_percentage_list.append(perturbed_sentence_percentage[token_idx])
                perturbed_word_percentage_list.append(perturbed_word_percentage[token_idx])
                perturbed_tokens_list.append(perturbed_tokens[token_idx])
        df_output = pd.DataFrame({'id': output_ids_list, 'reference_text': reference_text_list, 'perturbed_text': perturbed_text_list, 'perturbed_sentence_percentage': perturbed_sentence_percentage_list, 'perturbed_word_percentage': perturbed_word_percentage_list, 'perturbed_tokens': perturbed_tokens_list})
    elif args.task == 'add_definition':
        perturbed_word_n_list = []
        external_definitions_path = './external_knowledge_source/dbpedia.json'
        with open(external_definitions_path, 'r') as f:
            external_definitions = json.load(f)
        for i in tqdm(range(len(batch))):
            perturbed_text, perturbed_word_n, perturbed_tokens = template.perturb_iteration(batch[i], external_definitions, top_n=4)
            for token_idx in range(len(perturbed_text)):
                output_ids_list.append(ids[i])
                reference_text_list.append(batch[i])
                perturbed_text_list.append(perturbed_text[token_idx])
                perturbed_word_n_list.append(perturbed_word_n[token_idx])
                perturbed_tokens_list.append(perturbed_tokens[token_idx])
        df_output = pd.DataFrame({'id': output_ids_list, 'reference_text': reference_text_list, 'perturbed_text': perturbed_text_list, 'perturbed_word_n': perturbed_word_n_list, 'perturbed_tokens': perturbed_tokens_list})
        # keep if perturbed_word_n is not euqals to 0 (no definition found) or less than 4 (more than 3 definitions found)
        df_output = df_output[(df_output['perturbed_word_n'] != 0) & (df_output['perturbed_word_n'] < 4)]
        
    df_output.to_csv(os.path.join(args.output_path, args.task + '_' + args.input_file.split('.')[0] + '_perturbation.csv'), index=False)

if __name__ == '__main__':
    main()
