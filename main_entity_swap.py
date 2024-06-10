import os
import json
import pandas as pd
# from perturbation.factual_consistency_entity_swap import EntitySwap
from perturbation.factual_consistency_entity_swap import EntitySwap
import argparse
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='', help='path to the input file')
    parser.add_argument('--input_file', type=str, default='en_de_en_gpt3.csv', help='input file name')
    parser.add_argument('--output_path', type=str, default='', help='path to the output file')
    parser.add_argument('--task', type=str, default='entity_swap', help='number_swap, verb_swap, negate_sentence, coherent')
    
    args = parser.parse_args()
    
    df = pd.read_csv(args.input_path + args.input_file)
    # df = df[:2]
    batch = df['reference_text'].tolist()
    ids = df['id'].tolist()

    task_list = {   
                "entity_swap": EntitySwap()
            }

    template = task_list[args.task]

    output_ids_list = []
    reference_text_list = []
    perturbed_text_list = []
    perturbed_tokens_list = []

    if args.task == 'entity_swap':
        perturbed_word_n_list = []
        for i in tqdm(range(len(batch))):
            perturbed_text, perturbed_word_n, perturbed_tokens = template.perturb_iteration(batch[i])
            for token_idx in range(len(perturbed_text)):
                output_ids_list.append(ids[i])
                reference_text_list.append(batch[i])
                perturbed_text_list.append(perturbed_text[token_idx])
                perturbed_word_n_list.append(perturbed_word_n[token_idx])
                perturbed_tokens_list.append(perturbed_tokens[token_idx])
                # assert len(output_ids_list) == len(reference_text_list) == len(perturbed_text_list) == len(perturbed_word_n_list) == len(perturbed_tokens_list), 'lengths not equal'
        df_output = pd.DataFrame({'id': output_ids_list, 'reference_text': reference_text_list, 'perturbed_text': perturbed_text_list, 'perturbed_sentence_percentage': perturbed_word_n_list, 'perturbed_tokens': perturbed_tokens_list})
        
    df_output.to_csv(os.path.join(args.output_path, args.task + '_' + args.input_file.split('.')[0] + '_perturbation.csv'), index=False)

if __name__ == '__main__':
    main()
