

# 
for task in synonyms_verb_swap antonyms_verb_swap negate_sentence coherent delete_sentence add_non_related_sentence add_related_sentence add_definition simplification number_swap
do
      python gpt.py \
            --hypo_path ./src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive/permutation/back_translate_test_oracle_extractive_de/ \
            --hypo_file ${task}_en_de_en_gpt3_perturbation.csv \
            --id_file test_back_translate_gpt3_id30_for_eval.csv \
            --src_path ./src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive/ \
            --src_file test.source \
            --tgt_path ./src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive/ \
            --tgt_file test.target \
            --output_path ./src_100-1000_tgt_0-700_full_data_extract_elife_annals_medicine_reproductive/permutation/back_translate_test_oracle_extractive_de/gpt4/ \
            --criteria 'Informativeness, Simplification, Coherence, Faithfulness' \
            --model_function chat_gpt_no_ref_with_explain_all_criteria
done
