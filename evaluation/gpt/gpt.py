from utils import *
import pandas as pd
import os
import openai
from tqdm import tqdm
import json
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypo_path', type=str, default='./back_translate_test_oracle_extractive_de/', help='path to the input file')
    parser.add_argument('--hypo_file', type=str, default='delete_sentence_test_oracle_extractive_perturbation.csv', help='input file name')
    parser.add_argument('--id_file', type=str, default='test_back_translate_gpt3_id100_for_eval.csv', help='input file name')
    parser.add_argument('--src_path', type=str, default='./src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive/', help='path to the input file')
    parser.add_argument('--src_file', type=str, default='test.source', help='input file name')
    parser.add_argument('--tgt_path', type=str, default='./src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive/', help='path to the input file')
    parser.add_argument('--tgt_file', type=str, default='test.target', help='input file name')
    parser.add_argument('--output_path', type=str, default='./src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive/permutation/back_translate_test_oracle_extractive_de/gpt4/', help='path to the output file')
    parser.add_argument('--criteria', type=str, default='Informativeness')
    parser.add_argument('--model_function', type=str, default='chat_gpt_no_ref_no_explain')
    args = parser.parse_args()
    
    print('hypo_file: ', args.hypo_file)
    print('output_path: ', args.output_path)

    task_name = args.hypo_file.split('_en_de_en_gpt3_perturbation.csv')[0]

    with open(args.src_path + args.src_file, 'r') as f:
        src = f.readlines()
    with open(args.tgt_path + args.tgt_file, 'r') as f:
        tgt = f.readlines()
    
    df = pd.read_csv(args.hypo_path + args.hypo_file)

    # add src and tgt to df based on df['id']
    df['abs_txt'] = [src[i] for i in df['id']]
    df['tgt_txt'] = [tgt[i] for i in df['id']]
    # replace \n with space
    df['abs_txt'] = df['abs_txt'].apply(lambda x: x.replace('\n', ' '))
    df['tgt_txt'] = df['tgt_txt'].apply(lambda x: x.replace('\n', ' '))

    df_id = pd.read_csv(args.hypo_path + args.id_file)
    # select df based on df_id['id']
    df = df[df['id'].isin(df_id['id'])]

    criteria_list = args.criteria.split(',')
    criteria_list = [c[1:] if c[0] == ' ' else c for c in criteria_list]  # delete space in the beginning of the criteria
    criteria_prompt = generate_criteria_prompt(criteria_list)
    response_list = []

    model_function = globals()[args.model_function]
    # df = df[:1]
    if args.model_function == 'chat_gpt_with_ref_no_explain_all_criteria':
        for i, row in tqdm(df.iterrows(), total=len(df)):
            abstract = row['abs_txt']
            hypo = row['perturbed_text']
            ref = row['tgt_txt']
            try:
                response = model_function(abstract, hypo, ref, model_name='gpt-4', criteria=criteria_prompt)
                response_list.append(response)
            except Exception as e:
                print(f"Error occurred for abstract at index {i}: {str(e)}")
                response_list.append(None)
    else:
        for i, row in tqdm(df.iterrows(), total=len(df)):
            abstract = row['abs_txt']
            hypo = row['perturbed_text']
            try:
                response = model_function(abstract, hypo, model_name='gpt-4', criteria=criteria_prompt)
                response_list.append(response)
            except Exception as e:
                print(f"Error occurred for abstract at index {i}: {str(e)}")
                response_list.append(None)

    concate_criteria = '_'.join(criteria_list)
    df.loc[:, concate_criteria] = response_list

    # Check if the DataFrame has any rows
    if not df.empty:
        df.to_csv(args.output_path + f'{task_name}_{concate_criteria}_{args.model_function}.csv', index=False)
        # df_top_10.to_csv(args.output_path + f'baseline.csv', index=False)
    else:
        print("DataFrame is empty. No data to save.") 
    
if __name__ == '__main__':
    main()