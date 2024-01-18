import pandas as pd
import math
import numpy as np
import os

import argparse

def mean_std_fix(src, tgt, ppl_name='perplexity_score'):
    src['log_ppl'] = src[ppl_name].apply(lambda x: math.log(x))
    tgt['log_ppl'] = tgt[ppl_name].apply(lambda x: math.log(x))

    # calculate mean and std of ref_src_df and ref_tgt_df log_ppl
    log_ppl = src['log_ppl'].tolist() + tgt['log_ppl'].tolist()
    mean = np.mean(log_ppl)
    std = np.std(log_ppl)
    return mean, std

def z_score_fix(df, mean, std):
    zscore = df['log_ppl'].apply(lambda x: (x - mean) / std)
    return zscore

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_input_path', type=str, default='../data/', help='path to the input file')
    parser.add_argument('--indomain_model', type=str, default='pubmedgpt', help='indomain model name')
    parser.add_argument('--outdomain_model', type=str, default='pubmedgpt', help='outdomain model name')
    parser.add_argument('--ref_prefix_src', type=str, default='number_swap', help='number_swap, verb_swap, negate_sentence, coherent')
    parser.add_argument('--ref_prefix_tgt', type=str, default='number_swap', help='number_swap, verb_swap, negate_sentence, coherent')
    parser.add_argument('--hypo_input_path', type=str, default='../data/', help='path to the input file')
    parser.add_argument('--hypo_prefix', type=str, default='number_swap', help='number_swap, verb_swap, negate_sentence, coherent')
    parser.add_argument('--hypo_output_path', type=str, default='../output/', help='path to the output file')
    parser.add_argument('--ppl_name', type=str, default='perplexity_score', help='perplexity_score or ppl_score')
    args = parser.parse_args()

    # 1. get mean and std of indomain and outdomain
    ref_src_df_indomain = pd.read_csv(args.ref_input_path + args.ref_prefix_src + '_' + args.indomain_model + '_perplexity_score.csv')
    ref_tgt_df_indomain = pd.read_csv(args.ref_input_path + args.ref_prefix_tgt + '_' + args.indomain_model + '_perplexity_score.csv')
    ref_src_df_outdomain = pd.read_csv(args.ref_input_path + args.ref_prefix_src + '_' + args.outdomain_model + '_perplexity_score.csv')
    ref_tgt_df_outdomain = pd.read_csv(args.ref_input_path + args.ref_prefix_tgt + '_' + args.outdomain_model + '_perplexity_score.csv')
    
    indomain_mean, indomain_std = mean_std_fix(ref_src_df_indomain, ref_tgt_df_indomain, args.ppl_name)
    outdomain_mean, outdomain_std = mean_std_fix(ref_src_df_outdomain, ref_tgt_df_outdomain, args.ppl_name)
    
    # 2. get zscore of hypo
    hypo_df_indomain = pd.read_csv(args.hypo_input_path + args.hypo_prefix + '_' + args.indomain_model + '_perplexity_score.csv')
    hypo_df_outdomain = pd.read_csv(args.hypo_input_path + args.hypo_prefix + '_' + args.outdomain_model + '_perplexity_score.csv')
    
    # hypo_df = hypo_df.dropna()
    hypo_df_indomain['log_ppl'] = hypo_df_indomain[args.ppl_name].apply(lambda x: math.log(x))
    hypo_df_outdomain['log_ppl'] = hypo_df_outdomain[args.ppl_name].apply(lambda x: math.log(x))

    hypo_indomain_zscore = z_score_fix(hypo_df_indomain, indomain_mean, indomain_std)
    hypo_outdomain_zscore = z_score_fix(hypo_df_outdomain, outdomain_mean, outdomain_std)

    # 3. get plain score
    hypo_df_indomain[args.indomain_model] = hypo_indomain_zscore
    hypo_df_indomain[args.outdomain_model] = hypo_outdomain_zscore
    hypo_df_indomain['plain_score'] = hypo_df_indomain[args.indomain_model] - hypo_df_indomain[args.outdomain_model]
    hypo_df_indomain.to_csv(os.path.join(args.hypo_output_path, args.hypo_prefix + '_pomme_score.csv'), index=False)

    # 4. print hypo plain score
    print_dict = {}
    print_dict['indomain_model'] = args.indomain_model
    print_dict['outdomain_model'] = args.outdomain_model
    print_dict['hypo_prefix'] = args.hypo_prefix
    print_dict['hypo_mean'] = hypo_df_indomain[args.indomain_model].mean()
    print_dict['hypo_std'] = hypo_df_indomain[args.indomain_model].std()
    # print print_dict
    print('indomain_model: ', print_dict['indomain_model'])
    print('outdomain_model: ', print_dict['outdomain_model'])
    print('hypo_prefix: ', print_dict['hypo_prefix'])
    print('hypo_mean: ', print_dict['hypo_mean'])
    print('hypo_std: ', print_dict['hypo_std'])

if __name__ == '__main__':
    main()