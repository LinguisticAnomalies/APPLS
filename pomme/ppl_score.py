import numpy as np
import gc
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoTokenizer, OPTForCausalLM, BioGptTokenizer, BioGptForCausalLM, GPT2LMHeadModel, GPT2Tokenizer, AutoModelWithLMHead
import os
import argparse

def inputPerplexity(text, model, tokenizer):
    max_length = model.config.n_positions
    # max_length = 1024
    stride = 512
    encodings = tokenizer(text, return_tensors="pt", truncation=True)
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to("cuda")
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over input tokens.
            # Multiply it with trg_len to get the summation instead of average.
            # We will take average over all the tokens to get the true average
            # in the last step of this example.
            neg_log_likelihood = outputs.loss * trg_len

        nlls.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    return ppl.cpu().numpy()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hypo_path', type=str, default='../data/', help='path to the input file')
    parser.add_argument('--hypo_file', type=str, default='simplification_en_de_en_gpt3_perturbation.csv', help='input file name')
    parser.add_argument('--output_path', type=str, default='../output/', help='path to the output file')
    parser.add_argument('--model', type=str, default='facebook/opt-6.7b', help='model name')
    args = parser.parse_args()
    
    if '.csv' in args.hypo_file:
        df = pd.read_csv(args.hypo_path + args.hypo_file)
    else:
        with open (args.hypo_path + args.hypo_file, 'r') as f:
            perturbed_text_list = f.readlines()
        idx = [i for i in range(len(perturbed_text_list))]
        df = pd.DataFrame(zip(idx, perturbed_text_list), columns=['id', 'perturbed_text'])

    perturbed_text_list = df['perturbed_text'].tolist()
    id_list = df['id'].tolist()

    if args.model == 'opt':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b", use_fast=False, model_max_length=1024)
        model = OPTForCausalLM.from_pretrained("facebook/opt-2.7b", device_map="auto").cuda()
    elif args.model == 'galactica':
        tokenizer = AutoTokenizer.from_pretrained("facebook/galactica-1.3b", use_fast=False)
        model = OPTForCausalLM.from_pretrained("facebook/galactica-1.3b", device_map="auto").cuda()
    elif args.model == 'biogpt':
        tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt", use_fast=False)
        model = BioGptForCausalLM.from_pretrained("microsoft/biogpt").cuda()
    elif args.model == 'pubmedgpt':
        tokenizer = GPT2Tokenizer.from_pretrained("stanford-crfm/BioMedLM", use_fast=False, model_max_length=1024)
        model = GPT2LMHeadModel.from_pretrained("stanford-crfm/BioMedLM", device_map="auto").cuda()
    elif args.model == 't5':
        tokenizer = AutoTokenizer.from_pretrained("t5-3b", model_max_length=512)
        model = AutoModelWithLMHead.from_pretrained("t5-3b").cuda()  
    elif args.model == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2", use_fast=False, model_max_length=1024)
        model = GPT2LMHeadModel.from_pretrained("gpt2", device_map="auto").cuda()

    ppl_list = []
    for i in tqdm(range(len(perturbed_text_list))):
        idx = id_list[i]
        perturbed_text = perturbed_text_list[i]
        ppl = inputPerplexity(perturbed_text, model, tokenizer)
        ppl_list.append(ppl)
        gc.collect()
        torch.cuda.empty_cache()
    df['perplexity_score'] = ppl_list

    df.to_csv(os.path.join(args.output_path, args.hypo_file.split('.')[0] + '_' + args.model + '_perplexity_score.csv'), index=False)

if __name__ == '__main__':
    main()