import pandas as pd
from nltk.tokenize import sent_tokenize

path = '/edata/yguo50/plain_language/pls/data/src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive/permutation/back_translate_test_oracle_extractive_de/'
hypo_file = 'simplification_en_de_en_gpt3_perturbation.csv'
df = pd.read_csv(path + hypo_file)
df['index'] = df.index

# Tokenizing perturbed_text into individual sentences
df['sentences'] = df['perturbed_text'].apply(sent_tokenize)

# Expanding the sentences and keeping the original index
df_explode = df.explode('sentences').drop(columns='perturbed_text').rename(columns={'sentences': 'perturbed_text'})

print(df_explode.head(1))
